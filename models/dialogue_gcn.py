import torch
from transformers import *
from torch import nn
import numpy as np

class DialogueGCNC(nn.Module):

    def __init__(self, config):
        super(DialogueGCNCell, self).__init__()
        self.text_encoder = nn.GRU(config.text_in_dim, config.text_out_dim, bidirectional=True, batch_first=True)
        self.context_encoder = nn.GRU(config.context_in_dim, config.context_out_dim, bidirectional=True, batch_first=True)
#        self.audio_W = nn.Linear(config.audio_in_dim, config.audio_out_dim, bias=True)
#        self.vis_W = nn.Linear(config.vis_in_dim, config.vis_out_dim, bias=True)
        self.relu = torch.relu
        self.pred_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.suc_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.same_speak_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.diff_speak_rel = GraphConvolution(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.att_window_size = config.att_window_size
        self.edge_att_weights = nn.Linear(self.utt_embed_size, self.utt_embed_size, bias=False)
        self.w_aggr = nn.Parameter(self.utt_embed_size, self.utt_embed_size)

    def forward(self, x, adj):
        indept_embeds = self.embed_text(x)
        context_embeds = self.context_encoder(indept_embeds)[0]
        attn, relation_matrices = self.construct_edges_relations(context_embeds)
        pred_adj, suc_adj, same_speak_adj, diff_adj_matrix = relation_matrices
        h = self.pred_rel(x, adj)
        h += self.suc_rel(x, adj)
        h += self.same_speak_rel(x, adj)
        h += self.diff_speak_rel(x, adj)

        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def embed_text(self, text):
        # Input is a tensor of size N x L x G
        #   N - number of utterances
        #   L - length of longest (in # of words) utterance
        #   G - dimention of Glove embeddings
        return self.text_encoder(text)[0]

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
