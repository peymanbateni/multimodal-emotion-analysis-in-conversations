import torch
from transformers import *
from torch import nn

class DialogueGCNC(nn.Module):

    def __init__(self, config):
        super(DialogueGCNCell, self).__init__()
        self.text_encoder = nn.GRU(config.text_in_dim, config.text_out_dim, bidirectional=True, batch_first=True)
        self.context_encoder = nn.GRU(config.context_in_dim, config.context_out_dim, bidirectional=True, batch_first=True)
#        self.audio_W = nn.Linear(config.audio_in_dim, config.audio_out_dim, bias=True)
#        self.vis_W = nn.Linear(config.vis_in_dim, config.vis_out_dim, bias=True)
        self.relu = torch.relu
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.context_window_size = config.context_window_size
        self.edge_attn = nn.Linear(self.utt_embed_size, self.utt_embed_size)

    def forward(self, x, adj):
        indept_embeds = self.embed_text(x)
        context_embeds = self.context_encoder(indept_embeds)[0]
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def embed_text(self, text):
        # Input is a tensor of size N x L x G
        #   N - number of utterances
        #   L - length of longest (in # of words) utterance
        #   G - dimention of Glove embeddings
        return self.text_encoder(text)[0]

    def construct_edges(self, ut_embs, speaker ):
        # ut_embs is a tensor of size N x D
        #   N - number of utterances
        #   D - dimention of utterance embedding
        # speaker is a list of size N corresponding
        #   to speaker ids for each utterance
        N = len(ut_embs)
        for i in range(N):
            start_idx = max(i - self.edge_window, 0)
            end_idx = min(i + self.edge_window, N)
