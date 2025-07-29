from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import AttModel
from torch_geometric.data import Data
from .GCN_model import GCN

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def pad_tokens(att_feats):
    # ---->pad
    H = att_feats.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    att_feats = torch.cat([att_feats, att_feats[:, :add_length, :]], dim=1)  # [B, N, L]
    return att_feats

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

    def forward(self, src, tgt, src_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x,text_features,gcn_query, mask):
        for i,layer in enumerate(self.layers):
            x = layer(x,text_features,gcn_query, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn,self_attn2,self_attn3, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.self_attn2 = self_attn2
        self.self_attn3 = self_attn3
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.d_model = d_model

    def forward(self, x, text_features, gcn_query, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x_self =x
        x = self.sublayer[1](x, lambda x: self.self_attn2(x, gcn_query, gcn_query, mask))
        x = torch.cat([x, x_self], dim=1)
        x = self.self_attn3(text_features, x, x, mask)
        return self.sublayer[2](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PAM(nn.Module):
    def __init__(self, dim=512):
        super(PAM, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 13, 1, 13//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
    def forward(self, x):
        B, H, C = x.shape
        assert int(math.sqrt(H))**2==H, f'{x.shape}'
        cnn_feat = x.transpose(1, 2).view(B, C, int(math.sqrt(H)), int(math.sqrt(H))).contiguous()
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

class GCNRunner(nn.Module):
    def __init__(self, in_channels=None, hidden_channels=512, out_channels=200,
                 num_layers=2, dropout=0.5, use_pred=False, device=None):
        super(GCNRunner, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # initialize GCN
        self.gnn_model = GCN(in_channels=in_channels, hidden_channels=hidden_channels,
                             out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                             use_pred=use_pred).to(self.device)

    def forward(self, edges, node_features):
        """
        GCN
        """
        # Graph
        data = Data(x=node_features.to(self.device), edge_index=edges.to(self.device))
        out = self.gnn_model(data.x, data.edge_index)
        return out

class EncoderDecoder(AttModel):

    def make_model(self):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        encoder = Encoder(
            EncoderLayer(self.d_model, c(attn), c(attn), c(attn), c(ff), self.dropout),
            self.num_layers
        )

        # Initialization parameter
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return encoder


    def __init__(self, args):
        super(EncoderDecoder, self).__init__(args)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.num_classes = args.n_classes
        self.process_img_feat = nn.Linear(args.d_vf, args.d_model)
        self.t_l = nn.Parameter(torch.randn(2, args.d_model))
        self.attention_network = Attn_Net(L=768, D=256, dropout=True, n_classes=1)
        self.model = self.make_model()
        self.gcn_runner = GCNRunner(in_channels=768, hidden_channels=512, out_channels=768, num_layers=4, dropout=0.3)
        self.image_classifier = nn.Sequential(
            nn.Linear(args.d_model, 512),  # image feature's  Transformer output size is  d_model
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, self.num_classes)
        )
        
    def init_hidden(self, bsz):
        return []

    def compute_attention_scores(self, att_feats):
        """
        Compute attention scores for each patch using the attention network.
        """
        # Pass the patch features through the attention network (Attn_Net)
        attention_scores, _ = self.attention_network(att_feats)  # (patches_num, 1)
        attention_scores = attention_scores.squeeze(-1)  # Remove the extra dimension
        return attention_scores

    def select_patches_with_attention(self, att_feats, attention_scores, k_sample):
        """
        Select k_sample patches based on their attention scores for each sample in the batch.
        """
        # Select the top k_sample patches based on attention scores for each sample in the batch
        topk_values, topk_indices = torch.topk(attention_scores, k_sample, dim=1, largest=True)
        # sorted top-k
        sorted_indices, _ = torch.sort(topk_indices, dim=1)
        # Gather the selected patch features from att_feats
        selected_feats = torch.gather(att_feats, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, att_feats.size(2)))
        return selected_feats


    def build_spatial_graph(self, features, k_num=4):
        """
        - features:  (batch_size, num_nodes, d_model)
        - k_num: the number of KNN (k nearest neighbors)
        - edge_index: GCN's edge, (2, num_edges)
        - edge_weight: GCN's weight, (num_edges,)
        """
        batch_size, num_nodes, _ = features.shape
        edge_indices = []
        edge_weights = []
        for i in range(batch_size):
            batch_features = F.normalize(features[i], p=2, dim=-1)
            cos_sim = torch.mm(batch_features, batch_features.T)
            topk_values, topk_indices = torch.topk(cos_sim, k_num + 1, dim=-1)
            source_nodes = torch.arange(num_nodes).repeat_interleave(k_num).to(features.device)
            target_nodes = topk_indices[:, 1:].reshape(-1)
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_weight = topk_values[:, 1:].reshape(-1)
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)
        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = torch.cat(edge_weights)
        return edge_index, edge_weight

    def _prepare_feature_mesh(self, att_feats, att_masks=None, meshes=None):
        att_feats = pad_tokens(att_feats)
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.process_img_feat(att_feats)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True
            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None
        return att_feats, meshes, att_masks, meshes_mask


    def forward(self, att_feats, text_features, att_masks=None):
        att_feats, _, _, _ = self._prepare_feature_mesh(att_feats, att_masks)
        patch_num = att_feats.size(1)
        if patch_num > self.args.k_sample:
            attention_scores = self.compute_attention_scores(att_feats)
            selected_feats = self.select_patches_with_attention(att_feats, attention_scores,
                                                                k_sample=self.args.k_sample)
        else:
            selected_feats = att_feats
        t_l_expanded = self.t_l.expand(text_features.size(0), -1, -1)
        text_feature_with_t_l = torch.cat((t_l_expanded, text_features),dim=1)
        edge_index, edge_weight = self.build_spatial_graph(selected_feats)
        gcn_query = self.gcn_runner(edge_index, selected_feats.squeeze(0))
        out = self.model(selected_feats, text_feature_with_t_l, gcn_query.unsqueeze(0), att_masks)
        return out

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.long().unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]

    def _encode(self, fc_feats, att_feats, att_masks=None):
        att_feats, _, _, _ = self._prepare_feature_mesh(att_feats, att_masks)
        attention_scores = self.compute_attention_scores(att_feats)
        selected_feats = self.select_patches_with_attention(att_feats, attention_scores, k_sample=self.args.k_sample)
        out = self.model.encode(selected_feats,att_masks)
        return out