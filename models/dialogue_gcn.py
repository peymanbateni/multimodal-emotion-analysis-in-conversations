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
        self.text_encoder = nn.GRU(config.text_in_dim, config.text_out_dim, bidirectional=True, batch_first=True)
        self.context_encoder = nn.GRU(config.context_in_dim, config.context_out_dim, bidirectional=True, batch_first=True)
#        self.audio_W = nn.Linear(config.audio_in_dim, config.audio_out_dim, bias=True)
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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x):

        transcripts, video, audio, speakers = x
        speakers.squeeze_(0)
        indept_embeds = self.embed_text(transcripts)
        context_embeds = self.context_encoder(indept_embeds)[0].squeeze(0)
        relation_matrices = self.construct_edges_relations(context_embeds, speakers)
        pred_adj, suc_adj, same_speak_adj, diff_adj_matrix = relation_matrices

        h1 = self.pred_rel_l1(context_embeds, pred_adj)
        h1 += self.suc_rel_l1(context_embeds, suc_adj)
        h1 += self.same_speak_rel_l1(context_embeds, same_speak_adj)
        h1 += self.diff_speak_rel_l1(context_embeds, diff_adj_matrix)
        h1 = torch.relu(h1 + torch.matmul(context_embeds, self.w_aggr_1))

        h2 = self.pred_rel_l2(h1, pred_adj)
        h2 += self.suc_rel_l2(h1, suc_adj)
        h2 += self.same_speak_rel_l2(h1, same_speak_adj)
        h2 += self.diff_speak_rel_l2(h1, diff_adj_matrix)
        h2 = torch.relu(h2 + torch.matmul(h1, self.w_aggr_2))
        h = torch.cat([h2, context_embeds], dim=1)
        
        return h2

    def embed_text(self, texts):
        # Input is a tensor of size N x L x G
        #   N - number of utterances
        #   L - length of longest (in # of words) utterance
        #   G - dimention of Glove embeddings
        lengths = []
        for i, utt in enumerate(texts):
            # TODO: Fix UTT bug
            input_ids = torch.tensor([self.tokenizer.encode(utt[0])])
            all_hidden_states, all_attentions = self.bert(input_ids)[-2:]
            texts[i] = all_hidden_states.squeeze(0)
            lengths.append(all_hidden_states.size(1))
        texts = pad_sequence(texts, batch_first=True)
        lengths = torch.LongTensor(lengths)
        # Sort utterance transcripts in decreasing order by the number of words
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        texts = texts[sorted_idx]
        # Pack -> rnn -> unpack (to handle variable-length sequences)
        texts = pack_padded_sequence(texts, lengths=sorted_lengths, batch_first=True)
        encoded_text = self.text_encoder(texts)[1][0]
        #print(encoded_text.size())
        #padded_encoded_text, _ = pad_packed_sequence(encoded_text, batch_first=True)
        # Sorts back to original order
        _, orig_idx = sorted_idx.sort(0)
        encoded_text = encoded_text[orig_idx].unsqueeze(0)
        #padded_encoded_text = torch.index_select(padded_encoded_text, dim=1,index=lenths-1)
        return encoded_text

    def construct_edges_relations(self, ut_embs, speaker_ids):
        # ut_embs is a tensor of size N x D
        #   N - number of utterances
        #   D - dimention of utterance embedding
        # speaker is a list of size N corresponding
        #   to speaker ids for each utterance
        pad = torch.zeros(self.att_window_size, self.utt_embed_size*2)
        ut_embs_padded = torch.cat((pad, ut_embs, pad), 0)
        ut_embs_fat = torch.zeros(len(ut_embs), self.att_window_size*2+1, self.utt_embed_size*2)
        for i in range(len(ut_embs)):
            ut_embs_fat[i, :, :] = ut_embs_padded[i:i+self.att_window_size*2+1,:]
        raw_attn = self.edge_att_weights(ut_embs_fat)
        ut_embs = ut_embs.unsqueeze(2)
        raw_attn = torch.matmul(raw_attn, ut_embs)
        attn = torch.softmax(raw_attn, dim=1).squeeze(2)
        relation_matrices = self.build_relation_matrices(ut_embs, speaker_ids, attn)
        return relation_matrices

    def build_relation_matrices(self, ut_embs, speaker_ids, attn):
        num_utt = len(ut_embs)
        print("Number of utterances: ", num_utt)
        num_speakers = len(np.unique(speaker_ids))

        pred_adj = torch.ones(num_utt, num_utt, dtype=torch.long).triu(0)
        suc_adj = 1 - pred_adj.byte()
        same_adj_matrix = torch.zeros(num_utt, num_utt, dtype=torch.long)
        for i in range(num_speakers):
            same_speak_indices = speaker_ids == i
            same_adj_matrix[same_speak_indices] = same_speak_indices.long()
        diff_adj_matrix = 1 - same_adj_matrix.byte()
        # Masking out entries due to att_window_size
        attn_mask = torch.zeros(num_utt, num_utt)
        for j in range(num_utt):
            pred_adj[j, j+1+self.att_window_size:num_utt] = 0
            suc_adj[j, 0:max(0, j-self.att_window_size)] = 0
            same_adj_matrix[j, j+1+self.att_window_size:num_utt] = 0
            same_adj_matrix[j, 0:max(0, j-self.att_window_size)] = 0
            diff_adj_matrix[j, j+1+self.att_window_size:num_utt] = 0
            diff_adj_matrix[j, 0:max(0, j-self.att_window_size)] = 0
            left_attn_boundary = 0
            if (j-self.att_window_size < 0):
                left_attn_boundary = self.att_window_size - j
            right_attn_boundary = self.att_window_size*2 + 1
            if (j+self.att_window_size >= num_utt):
                right_attn_boundary = right_attn_boundary - (j+self.att_window_size - num_utt + 1)
            #print("attn ", j)
            left_mask_boundary = max(0, j-self.att_window_size)
            right_mask_boundary = min(num_utt, j+self.att_window_size+1)
            #print(attn[j, left_attn_boundary:right_attn_boundary].size(), left_attn_boundary, right_attn_boundary)
            #print(attn_mask[j, left_mask_boundary:right_mask_boundary].size(),
            #    left_mask_boundary,
            #    right_mask_boundary)
            attn_mask[j, left_mask_boundary:right_mask_boundary] *= attn[j, left_attn_boundary:right_attn_boundary]

        pred_adj = pred_adj.float() * attn_mask
        suc_adj = suc_adj.float() * attn_mask
        same_adj_matrix = same_adj_matrix.float() * attn_mask
        diff_adj_matrix = diff_adj_matrix.float() * attn_mask
        return pred_adj, suc_adj, same_adj_matrix, diff_adj_matrix
