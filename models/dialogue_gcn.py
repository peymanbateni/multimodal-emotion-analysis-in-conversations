import torch
from torch import nn
import numpy as np
from models.dialogue_gcn_cell import GraphConvolution
from torch.nn.parameter import Parameter
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence


class DialogueGCN(nn.Module):

    def __init__(self, config):
        super(DialogueGCN, self).__init__()

        self.att_window_size = config.att_window_size
        self.utt_embed_size = config.utt_embed_size
        self.text_encoder = nn.GRU(config.text_in_dim, config.text_out_dim, batch_first=True)
        self.context_encoder = nn.GRU(config.context_in_dim, config.context_out_dim, bidirectional=True, batch_first=True)#        self.audio_W = nn.Linear(config.audio_in_dim, config.audio_out_dim, bias=True)
#        self.vis_W = nn.Linear(config.vis_in_dim, config.vis_out_dim, bias=True)
        
        self.pred_rel_l1 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.suc_rel_l1 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.same_speak_rel_l1 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.diff_speak_rel_l1 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        
        self.pred_rel_l2 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.suc_rel_l2 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.same_speak_rel_l2 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.diff_speak_rel_l2 = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.edge_att_weights = nn.Linear(self.utt_embed_size*2, self.utt_embed_size*2, bias=False)
        self.w_aggr_1 = nn.Parameter(torch.FloatTensor(self.utt_embed_size*2, self.utt_embed_size*2))
        self.w_aggr_2 = nn.Parameter(torch.FloatTensor(self.utt_embed_size*2, self.utt_embed_size*2))
        
        self.w_sentiment = nn.Linear(self.utt_embed_size*4, 3)
        self.w_emotion_1 = nn.Linear(self.utt_embed_size*4, self.utt_embed_size*2)
        self.w_emotion_2 = nn.Linear(self.utt_embed_size*2, 7)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x):
        transcripts, video, audio, speakers = x
        speakers.squeeze_(0)
        #print(self.pred_rel_l1.weight[300:350, 300:400])
        indept_embeds = self.embed_text(transcripts)
        context_embeds = self.context_encoder(indept_embeds)[0].squeeze(0)
        relation_matrices = self.construct_edges_relations(context_embeds, speakers)
        pred_adj, suc_adj, same_speak_adj, diff_adj_matrix, attn = relation_matrices
        #print(context_embeds[:, 300:330])                                        
        h1 = self.pred_rel_l1(context_embeds, pred_adj)
        #print(h1[:, 300:330])                                        
        h1 += self.suc_rel_l1(context_embeds, suc_adj)
        h1 += self.same_speak_rel_l1(context_embeds, same_speak_adj)
        h1 += self.diff_speak_rel_l1(context_embeds, diff_adj_matrix)
        h1 = torch.relu(h1 + torch.matmul(context_embeds, self.w_aggr_1) * attn.diag().unsqueeze(1))
        
        h2 = self.pred_rel_l2(h1, pred_adj)
        h2 += self.suc_rel_l2(h1, suc_adj)
        h2 += self.same_speak_rel_l2(h1, same_speak_adj)
        h2 += self.diff_speak_rel_l2(h1, diff_adj_matrix)
        h2 = torch.relu(h2 + torch.matmul(h1, self.w_aggr_2) * attn.diag().unsqueeze(1))
        h = torch.cat([h2, context_embeds], dim=1)
        #print(h2[:, 0:30])                                
        return self.w_emotion_2(torch.relu(self.w_emotion_1(h))), self.w_sentiment(h)

    def embed_text(self, texts):
        # Input is a tensor of size N x L x G
        #   N - number of utterances
        #   L - length of longest (in # of words) utterance
        #   G - dimention of Glove embeddings
        lengths = []
        for i, utt in enumerate(texts):
            input_ids = torch.tensor([self.tokenizer.encode(utt[0])]).to("cuda")
            hidden_states = self.bert(input_ids)[0]
            texts[i] = hidden_states.squeeze(0)
            lengths.append(hidden_states.size(1))
        texts = pad_sequence(texts, batch_first=True)
        lengths = torch.LongTensor(lengths)
        # Sort utterance transcripts in decreasing order by the number of words
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        texts = texts[sorted_idx]
        # Pack -> rnn -> unpack (to handle variable-length sequences)
        texts = pack_padded_sequence(texts, lengths=sorted_lengths, batch_first=True)
        encoded_text = self.text_encoder(texts)[1][0]
        # Sorts back to original order
        _, orig_idx = sorted_idx.sort(0)
        encoded_text = encoded_text[orig_idx].unsqueeze(0)
        return encoded_text    
        
    def construct_edges_relations(self, ut_embs, speaker_ids):
        # ut_embs is a tensor of size N x D
        #   N - number of utterances
        #   D - dimention of utterance embedding
        # speaker is a list of size N corresponding
        #   to speaker ids for each utterance
        num_utts = len(ut_embs)
        raw_attn = self.edge_att_weights(ut_embs)
        attn = torch.zeros(num_utts, num_utts).to("cuda")
        for i in range(num_utts):
            curr_utt = ut_embs[i]
            left_bdry = max(0, i - self.att_window_size)
            right_bdry = min(num_utts, i + self.att_window_size + 1)
            for j in range(left_bdry, right_bdry):
                attn[i, j] = curr_utt.dot(ut_embs[j])
        # TODO: Problematic: zero elements pull nonzero attention weights
        attn = torch.softmax(attn, dim=1)
        relation_matrices = self.build_relation_matrices(ut_embs, speaker_ids, attn)
        return relation_matrices
    
    """
    def construct_edges_relations(self, ut_embs, speaker_ids):
        # ut_embs is a tensor of size N x D
        #   N - number of utterances
        #   D - dimention of utterance embedding
        # speaker is a list of size N corresponding
        #   to speaker ids for each utterance
        pad = torch.zeros(self.att_window_size, self.utt_embed_size*2).to("cuda")
        ut_embs_padded = torch.cat((pad, ut_embs, pad), 0)
        
        ut_embs_fat = torch.zeros(len(ut_embs), self.att_window_size*2+1, self.utt_embed_size*2).to("cuda")
        for i in range(len(ut_embs)):
            ut_embs_fat[i, :, :] = ut_embs_padded[i:i+self.att_window_size*2+1,:]
#        print(ut_embs_fat[0, self.att_window_size])
  #      print(self.edge_att_weights.weight[300:350, 0:350])
        raw_attn = self.edge_att_weights(ut_embs_fat)
 #       print(raw_attn.size())
        ut_embs = ut_embs.unsqueeze(2)
        raw_attn = torch.matmul(raw_attn, ut_embs).squeeze(2)
        # TODO: Problematic: zero elements pull nonzero attention weights
        #attn = torch.softmax(raw_attn, dim=1)
        relation_matrices = self.build_relation_matrices(ut_embs, speaker_ids, raw_attn)
        return relation_matrices
    """
    def build_relation_matrices(self, ut_embs, speaker_ids, attn_mask):
        num_utt = len(ut_embs)
        #print("Number of utterances: ", num_utt)
        num_speakers = len(np.unique(speaker_ids))
        pred_adj = torch.ones(num_utt, num_utt, dtype=torch.long).triu(0).to("cuda")
        suc_adj = 1 - pred_adj.byte()
        same_adj_matrix = torch.zeros(num_utt, num_utt, dtype=torch.long).to("cuda")
        for i in range(num_speakers):
            same_speak_indices = speaker_ids == i
            same_adj_matrix[same_speak_indices] = same_speak_indices.long().to("cuda")
        diff_adj_matrix = 1 - same_adj_matrix.byte()
        pred_adj = pred_adj.float() * attn_mask
        suc_adj = suc_adj.float() * attn_mask
        same_adj_matrix = same_adj_matrix.float() * attn_mask
        diff_adj_matrix = diff_adj_matrix.float() * attn_mask
        return pred_adj, suc_adj, same_adj_matrix, diff_adj_matrix, attn_mask
    """
    def build_relation_matrices(self, ut_embs, speaker_ids, attn):
        num_utt = len(ut_embs)
        #print("Number of utterances: ", num_utt)
        num_speakers = len(np.unique(speaker_ids))
        pred_adj = torch.ones(num_utt, num_utt, dtype=torch.long).triu(0).to("cuda")
        suc_adj = 1 - pred_adj.byte()
        same_adj_matrix = torch.zeros(num_utt, num_utt, dtype=torch.long).to("cuda")
        for i in range(num_speakers):
            same_speak_indices = speaker_ids == i
            same_adj_matrix[same_speak_indices] = same_speak_indices.long().to("cuda")
        diff_adj_matrix = 1 - same_adj_matrix.byte()
        # Masking out entries due to att_window_size
        attn_mask = torch.zeros(num_utt, num_utt).to("cuda")
        for j in range(num_utt):
            left_attn_boundary = 0
            if (j-self.att_window_size < 0):
                left_attn_boundary = self.att_window_size - j
            right_attn_boundary = self.att_window_size*2 + 1
            if (j+self.att_window_size >= num_utt):
                right_attn_boundary = right_attn_boundary - (j+self.att_window_size - num_utt + 1)
            left_mask_boundary = max(0, j-self.att_window_size)
            right_mask_boundary = min(num_utt, j+self.att_window_size+1)
            #print(attn[j, left_attn_boundary:right_attn_boundary])
            attn_mask[j, left_mask_boundary:right_mask_boundary] += torch.softmax(attn[j, left_attn_boundary:right_attn_boundary], dim=0)
        # +1 for the case when there was only 1 speaker in the whole dialogue
        pred_adj = (pred_adj.float() / (torch.sum(pred_adj, dim=1) + 1)) * attn_mask
        suc_adj = (suc_adj.float() / (torch.sum(suc_adj, dim=1) + 1)) * attn_mask
        same_adj_matrix = (same_adj_matrix.float() / (torch.sum(same_adj_matrix, dim=1) + 1)) * attn_mask
        diff_adj_matrix = (diff_adj_matrix.float() / (torch.sum(diff_adj_matrix, dim=1) + 1)) * attn_mask
        return pred_adj, suc_adj, same_adj_matrix, diff_adj_matrix, attn_mask
    """
