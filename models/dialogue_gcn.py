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
        self.context_encoder = nn.GRU(config.context_in_dim * 2, config.context_out_dim, bidirectional=True, batch_first=True)
#        self.audio_W = nn.Linear(config.audio_in_dim, config.audio_out_dim, bias=True)
#        self.vis_W = nn.Linear(config.vis_in_dim, config.vis_out_dim, bias=True)
        self.relu = torch.relu
        self.pred_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.suc_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.same_speak_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.diff_speak_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.edge_att_weights = nn.Linear(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.w_aggr = nn.Parameter(torch.FloatTensor(self.utt_embed_size, self.utt_embed_size))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        transcripts, video, audio, speakers = x
        indept_embeds = self.embed_text(transcripts)
        print(indept_embeds[0].size())
        context_embeds = self.context_encoder(indept_embeds)[0]
        attn, relation_matrices = self.construct_edges_relations(context_embeds, speakers)
        pred_adj, suc_adj, same_speak_adj, diff_adj_matrix = relation_matrices
        h = self.pred_rel(x, adj)
        h += self.suc_rel(x, adj)
        h += self.same_speak_rel(x, adj)
        h += self.diff_speak_rel(x, adj)
        h = torch.relu(h + torch.matmul(raw_attn, context_embeds))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

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
            lengths.append(len(all_hidden_states))
        texts = pad_sequence(texts, batch_first=True)
        lengths = torch.LongTensor(lengths)
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        texts = texts[sorted_idx]
        print(texts.size())
        texts = pack_padded_sequence(texts, lengths=lengths, batch_first=True)
        encoded_text = self.text_encoder(texts)[0]
        return pad_packed_sequence(encoded_text, batch_first=True)

    def construct_edges_relations(self, ut_embs, speaker_ids):
        # ut_embs is a tensor of size N x D
        #   N - number of utterances
        #   D - dimention of utterance embedding
        # speaker is a list of size N corresponding
        #   to speaker ids for each utterance
        pad = torch.zeros(self.att_window_size, self.utt_embed_size)
        ut_embs_padded = torch.cat((pad, ut_embs, pad), 0)
        ut_embs_fat = torch.zeros(len(ut_embs), 2 * self.att_window_size, self.utt_embed_size)
        for i in range(len(ut_embs)):
            ut_embs_fat[i, :, :] = ut_embs_padded[i+self.att_window_size:i+self.att_window_size*2,:]
        raw_attn = self.edge_att_weights(ut_embs_fat)
        ut_embs = ut_embs.unsqueeze(2)
        raw_attn = torch.matmul(raw_attn, ut_embs)
        attn = torch.softmax(raw_attn, dim=1)
        relation_matrices = self.build_relation_matrices(ut_embs, speaker_ids, attn)
        return relation_matrices

    def build_relation_matrices(self, ut_embs, speaker_ids):
        num_utt = len(ut_embs)
        num_speakers = len(np.unique(speaker_ids))

        pred_adj = torch.ones(num_utt, num_utt).triu(0)
        suc_adj = 1 - pred_adj.byte()
        same_adj_matrix = torch.zeros(num_utt, num_utt)
        for i in range(num_speakers):
            same_speak_indices = speaker_ids == i
            same_adj_matrix[same_speak_indices] = same_speak_indices.long()
        diff_adj_matrix = 1 - same_adj_matrix.byte()
        # Masking out entries due to att_window_size
        for j in range(num_utt):
            pred_adj[j, j+1+self.att_window_size:num_utt] = 0
            suc_adj[j, 0:max(0, j-self.att_window_size)] = 0
            same_adj_matrix[j, j+1+self.att_window_size:num_utt] = 0
            same_adj_matrix[j, 0:max(0, j-self.att_window_size)] = 0
            diff_adj_matrix[j, j+1+self.att_window_size:num_utt] = 0
            diff_adj_matrix[j, 0:max(0, j-self.att_window_size)] = 0
        pred_adj *= attn
        suc_adj *= attn
        same_adj_matrix *= attn
        diff_adj_matrix *= attn
        return pred_adj, suc_adj, same_adj_matrix.long(), diff_adj_matrix.long()
