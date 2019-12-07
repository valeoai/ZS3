import torch
from pygcn.layers import GraphConvolution
from torch import nn


class GMMNnetwork(nn.Module):
    def __init__(
        self,
        noise_dim,
        embed_dim,
        hidden_size,
        feature_dim,
        semantic_reconstruction=False,
    ):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.5))
            return layers

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if hidden_size:
            self.model = nn.Sequential(
                *block(noise_dim + embed_dim, hidden_size),
                nn.Linear(hidden_size, feature_dim),
            )
        else:
            self.model = nn.Linear(noise_dim + embed_dim, feature_dim)

        self.model.apply(init_weights)
        self.semantic_reconstruction = semantic_reconstruction
        if self.semantic_reconstruction:
            self.semantic_reconstruction_layer = nn.Linear(
                feature_dim, noise_dim + embed_dim
            )

    def forward(self, embd, noise):
        features = self.model(torch.cat((embd, noise), 1))
        if self.semantic_reconstruction:
            semantic = self.semantic_reconstruction_layer(features)
            return features, semantic
        else:
            return features


class GMMNnetwork_GCN(nn.Module):
    def __init__(self, noise_dim=300, embed_dim=300, hidden_size=256, feature_dim=256):
        super().__init__()
        self.gcn1 = GraphConvolution(noise_dim + embed_dim, hidden_size)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.gcn2 = GraphConvolution(hidden_size, feature_dim)
        for m in self.modules():
            if isinstance(m, GraphConvolution):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, embd, noise, adj_mat):
        x = self.gcn1(torch.cat((embd, noise), 1), adj_mat)
        x = self.dropout(self.relu(x))
        return self.gcn2(x, adj_mat)
