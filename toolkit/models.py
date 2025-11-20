from .base import MLPBlock
from torch import nn
import torch as th
from dgl.nn.pytorch import GINConv, GraphConv
import torch.nn.functional as F


class GMSA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, feat_drop, att_drop):
        super(GMSA, self).__init__()
        self.gcn1 = GraphConv(input_dim, hidden_dim, weight=True, activation=F.relu, allow_zero_in_degree=True)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim, weight=True, activation=F.relu, allow_zero_in_degree=True)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.featdrop = nn.Dropout(feat_drop)
        self.attdrop = nn.Dropout(att_drop)

    def forward(self, blocks):
        Q = self.featdrop(self.gcn1(blocks[0], blocks[0].ndata['feat']['_N']))
        K = self.featdrop(self.gcn1(blocks[0], blocks[0].ndata['feat']['_N']))
        V = self.featdrop(self.gcn1(blocks[0], blocks[0].ndata['feat']['_N']))

        batch_size = blocks[1].ndata['feat']['_N'].size(0)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        attention_scores = th.einsum("bhd,bhd->bh", Q, K)
        attention_scores = F.softmax(attention_scores / self.head_dim ** 0.5, dim=-1)
        attention_scores = self.attdrop(attention_scores)
        attended_values = th.einsum("bh,bhd->bhd", attention_scores, V)
        attended_values = attended_values.reshape(batch_size, -1)
        output = self.gcn2(blocks[1], attended_values)
        output = self.norm(output)
        return output



class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.residual_projection = (
            nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        )
    def forward(self, x):
        residual = self.residual_projection(x)
        hidden = self.input_layer(x)
        hidden = hidden + residual
        output = self.output_layer(hidden)
        return output


class GINModel(nn.Module):
    def __init__(self, input_length, nb_gos, device, hidden_dim=1024, embed_dim=1024, dropout=0.3):
        super().__init__()
        self.li1 = nn.Sequential(
            nn.Linear(input_length, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.li2 = nn.Sequential(
            nn.Linear(input_length+hidden_dim, embed_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.af = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.1, batch_first=True)

        self.conv0 = GINConv(self.li1, 'max', learn_eps=True, activation=F.relu)
        self.conv1 = GINConv(self.li2, 'max', learn_eps=True, activation=F.relu)
        self.attBlock = GMSA(input_length, hidden_dim, num_heads=8, feat_drop=0.1, att_drop=0.5)
        self.fc_go_increse_dim = MLPBlock(128, hidden_dim)
        self.res_net = ResNet(hidden_dim, embed_dim, nb_gos)
        self.net2 = nn.Sequential(
            nn.Linear(embed_dim, nb_gos),
            nn.Sigmoid()
        )

    def forward(self, input_nodes, output_nodes, blocks, graph, go_emb):
        x0 = blocks[0].ndata['feat']['_N']
        x1 = self.conv0(blocks[0], x0)
        bs = x1.size(0)
        x1 = th.cat((x1, blocks[1].ndata['feat']['_N']), dim=-1)
        x2 = self.conv1(blocks[1], x1)
        x2_a = self.attBlock(blocks)
        go_emb = go_emb.expand(x2.size(0), -1)
        go_emb = self.fc_go_increse_dim(go_emb)
        x_all = x2 + x2_a + go_emb
        logits = self.res_net(x_all)
        return logits

