import torch
from torch import nn
import numpy as np
from models.dialogue_gcn_cell import GraphConvolution
from torch.nn.parameter import Parameter
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from models.visual_features import FaceModule
from models.visual_model import ExpressionDetector

class DialogueGCN(nn.Module):

    def __init__(self, config):
        super(DialogueGCN, self).__init__()
        self.config = config
        self.use_our_audio = config.use_our_audio
        self.use_meld_audio = config.use_meld_audio
        self.use_visual = config.use_visual
        self.att_window_size = config.att_window_size
        self.utt_embed_size = config.utt_embed_size
        self.text_encoder = nn.GRU(config.text_in_dim, config.text_out_dim, bidirectional=True, batch_first=True)
        if self.use_meld_audio:
            if self.use_visual:
                self.context_encoder = nn.GRU(config.context_in_dim * 4, config.context_out_dim, bidirectional=True, batch_first=True)#
            else:
                self.context_encoder = nn.GRU(config.context_in_dim * 3, config.context_out_dim, bidirectional=True, batch_first=True)#        
        else:
            if self.use_visual:
                self.context_encoder = nn.GRU(config.context_in_dim * 3, config.context_out_dim, bidirectional=True, batch_first=True)#
            else:
                self.context_encoder = nn.GRU(config.context_in_dim * 2, config.context_out_dim, bidirectional=True, batch_first=True)#        
        self.audio_W = nn.Linear(config.audio_in_dim, config.audio_out_dim, bias=True)
        #if self.use_meld_audio or self.use_our_audio:
        #    if self.config.use_texts:
        
        #self.context_encoder = nn.GRU(config.context_in_dim * 3, config.context_out_dim, bidirectional=True, batch_first=True)
        #    else:
        #        self.context_encoder = nn.GRU(config.context_in_dim * 1, config.context_out_dim, bidirectional=True, batch_first=True)
        #    self.audio_rnn = nn.GRU(41, 50, bidirectional=True, batch_first=True)
        #    self.audio_W = nn.Linear(1422, config.audio_out_dim, bias=True)
        #    self.audio_attn = nn.Linear(100, 1, bias=False)
        #    self.conv1_mel = nn.Conv2d(1, 30, 3, stride=2, padding=2)
        #    self.conv2_mel = nn.Conv2d(30 , 1, 3, stride=2, padding=2)
        #    self.conv1_mfcc = nn.Conv2d(1, 30, 3, stride=2,padding=2)
        #    self.conv2_mfcc = nn.Conv2d(30 , 1, 3, stride=2, padding=2)            
        #else:
        #    self.context_encoder = nn.GRU(config.context_in_dim * 2, config.context_out_dim, bidirectional=True, batch_first=True)#
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
        
        self.text_attn = nn.Linear(self.utt_embed_size * 2, 1)
        self.w_sentiment = nn.Linear(self.utt_embed_size*4, 3)
        self.w_emotion_1 = nn.Linear(self.utt_embed_size*4, self.utt_embed_size*2)
        self.w_emotion_2 = nn.Linear(self.utt_embed_size*2, 7)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.visual_module = ExpressionDetector(config.fan_weights_path)

    def forward(self, x):
        transcripts, video, audio, speakers = x
        speakers.squeeze_(0)
        #print(self.pred_rel_l1.weight[300:350, 300:400])
        if self.config.use_texts:
            indept_embeds = self.embed_text(transcripts)
        if self.use_meld_audio:
            audio = torch.stack(audio, dim=1).float().to('cuda')
            audio = torch.relu(self.w_reduce(audio))        
            indept_embeds = torch.cat([indept_embeds, audio], dim=2)
        if self.use_visual:
            visual_embeds, _ = self.visual_module(video)
            indept_embeds = torch.cat([indept_embeds, visual_embeds.unsqueeze(0)], dim=2)
            #audio = torch.relu(self.audio_W(audio))      
            #if self.config.use_texts:
            #    indept_embeds = torch.cat([indept_embeds, audio], dim=2)
            #else:
            #    indept_embeds = audio            
        #elif self.use_our_audio:
        #    audio = self.embed_audio(audio)
        #    if self.config.use_texts:
        #        indept_embeds = torch.cat([indept_embeds, audio], dim=2)
        #    else:
        #        indept_embeds = audio
                
        context_embeds = self.context_encoder(indept_embeds)[0].squeeze(0)
        #face_embeds = self.face_module(video)
        relation_matrices = self.construct_edges_relations(context_embeds, speakers)
        pred_adj, suc_adj, same_speak_adj, diff_adj_matrix, attn = relation_matrices
        #print(context_embeds[:, 300:330])  
        
        h1 = self.pred_rel_l1(context_embeds, pred_adj)
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
        
        return self.w_emotion_2(torch.relu(self.w_emotion_1(h))), self.w_sentiment(h)

    def embed_text(self, utterances):
        # Input is a tensor of size N x L x G
        #   N - number of utterances
        #   L - length of longest (in # of words) utterance
        #   G - dimention of Glove embeddings
        lengths = []
        texts = []
        for i, utt in enumerate(utterances):
            input_ids = torch.tensor([self.tokenizer.encode(utt[0])]).to("cuda")
            hidden_states = self.bert(input_ids)[0]
            texts.append(hidden_states.squeeze(0))
            lengths.append(hidden_states.size(1))
        texts = pad_sequence(texts, batch_first=True)
        lengths = torch.LongTensor(lengths)
        # Sort utterance transcripts in decreasing order by the number of words
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        texts = texts[sorted_idx]
        # Pack -> rnn -> unpack (to handle variable-length sequences)
        texts = pack_padded_sequence(texts, lengths=sorted_lengths, batch_first=True)
        encoded_text = pad_packed_sequence(self.text_encoder(texts)[0], batch_first=True)[0]
        # Attention over word sequences
        averaged_vectors = []
        for i, seq in enumerate(encoded_text):
            word_vecs = seq[0:sorted_lengths[i], :]
            text_raw_attn = self.text_attn(word_vecs)
            text_attn = torch.softmax(text_raw_attn, dim=0)
            att_vec = word_vecs * text_attn
            #print(att_vec, att_vec.sum(dim=0).size())
            averaged_vectors.append(att_vec.sum(dim=0))
        encoded_text = torch.stack(averaged_vectors, dim=0)
        # Sorts back to original order
        _, orig_idx = sorted_idx.sort(0)
        encoded_text = encoded_text[orig_idx].unsqueeze(0)
        return encoded_text    

    def embed_audio(self, audio):
        lengths = []
        audios = []
#        print("audio size ", len(audio))
        for i, utt in enumerate(audio):
            melspec, mfcc = utt[0], utt[1]
            melspec = melspec.reshape(melspec.size(-1), -1).to('cuda').unsqueeze(0).unsqueeze(0)
            mfcc = mfcc.reshape(mfcc.size(-1), -1).to('cuda').unsqueeze(0).unsqueeze(0)
            melspec = self.conv2_mel(torch.relu(self.conv1_mel(melspec))).squeeze(0).squeeze(0)
            mfcc = self.conv2_mfcc(torch.relu(self.conv1_mfcc(mfcc))).squeeze(0).squeeze(0)
            audio_feat = torch.cat([melspec, mfcc], dim=1)
            audios.append(audio_feat)
            lengths.append(audio_feat.size(0))
        audios = pad_sequence(audios, batch_first=True)
        lengths = torch.LongTensor(lengths)
        # Sort utterance transcripts in decreasing order by the number of words
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        audios = audios[sorted_idx]
        # Pack -> rnn -> unpack (to handle variable-length sequences)
        audios = pack_padded_sequence(audios, lengths=sorted_lengths, batch_first=True)
        encoded_audio = pad_packed_sequence(self.audio_rnn(audios)[0], batch_first=True)[0]
        # Attention over word sequences
        averaged_vectors = []
        for i, seq in enumerate(encoded_audio):
            audio_vecs = seq[0:sorted_lengths[i], :]
            audio_raw_attn = self.audio_attn(audio_vecs)
            audio_attn = torch.softmax(audio_raw_attn, dim=0)
            att_vec = audio_vecs * audio_attn
            #print(att_vec, att_vec.sum(dim=0).size())
            averaged_vectors.append(att_vec.sum(dim=0))
        encoded_audio = torch.stack(averaged_vectors, dim=0)
        # Sorts back to original order
        _, orig_idx = sorted_idx.sort(0)
        encoded_audio = encoded_audio[orig_idx].unsqueeze(0)
        #print(encoded_audio.size())
        return encoded_audio    
    
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
    
    def build_relation_matrices(self, ut_embs, speaker_ids, attn_mask):
        num_utt = len(ut_embs)
        #print("Number of utterances: ", num_utt)
        num_speakers = len(np.unique(speaker_ids))
        pred_adj = torch.ones(num_utt, num_utt, dtype=torch.float).triu(0).to("cuda")
        suc_adj = 1 - pred_adj
        same_adj_matrix = torch.zeros(num_utt, num_utt, dtype=torch.long).to("cuda")
        for i in range(num_speakers):
            same_speak_indices = speaker_ids == i
            same_adj_matrix[same_speak_indices] = same_speak_indices.long().to("cuda")
        diff_adj_matrix = 1 - same_adj_matrix.byte()
        same_adj_pred = same_adj_matrix.float() * pred_adj.float() * attn_mask
        same_adj_post = same_adj_matrix.float() * suc_adj.float() * attn_mask
        diff_adj_pred = diff_adj_matrix.float() * pred_adj.float() * attn_mask
        diff_adj_post = diff_adj_matrix.float() * suc_adj.float() * attn_mask
        return same_adj_pred, same_adj_post, diff_adj_pred, diff_adj_post, attn_mask
