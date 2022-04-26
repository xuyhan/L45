import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9
from torch_scatter import scatter


class MPNNLayerNoBN(MessagePassing):
    def __init__(self, emb_dim=64, aggr='max'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim, emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), ReLU()
        )
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), ReLU()
        )

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j):
        return self.mlp_msg(torch.cat([h_i, h_j], dim=-1))

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        return self.mlp_upd(torch.cat([h, aggr_out], dim=-1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNModelNoBN(Module):
    def __init__(self, num_layers=5, emb_dim=64, in_dim=6, out_dim=1):
        super().__init__()
        self.lin_in = Linear(in_dim, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, aggr='max'))
        self.pool = global_mean_pool
        self.lin_pred = Linear(emb_dim, out_dim)
        self.mlp = torch.nn.Sequential(
            Linear(emb_dim * 2, 32),
            ReLU(),
            Linear(32, 1)
        )

    def forward(self, x, edge_index):
        h = self.lin_in(x)
        n_nodes = x.shape[0]
        for conv in self.convs:
            h = h + conv(h, edge_index)
        us, vs = edge_index
        h = torch.column_stack([h.index_select(0, us), h.index_select(0, vs)])
        h = self.mlp(h).squeeze()
        E = h.new_zeros((n_nodes, n_nodes))
        E[tuple(edge_index)] += h
        return E


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='max'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j):
        return self.mlp_msg(torch.cat([h_i, h_j], dim=-1))

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        return self.mlp_upd(torch.cat([h, aggr_out], dim=-1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNModel(Module):
    def __init__(self, num_layers=5, emb_dim=64, in_dim=6, out_dim=1):
        super().__init__()
        self.lin_in = Linear(in_dim, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, aggr='max'))
        self.pool = global_mean_pool
        self.lin_pred = Linear(emb_dim, out_dim)
        self.mlp = torch.nn.Sequential(
            Linear(emb_dim * 2, 32),
            ReLU(),
            Linear(32, 1)
        )
        self.bn = BatchNorm1d(emb_dim)

    def forward(self, x, edge_index):
        h = self.lin_in(x)
        n_nodes = x.shape[0]
        for conv in self.convs:
            h = self.bn(h)
            h = h + conv(h, edge_index)
        us, vs = edge_index
        h = torch.column_stack([h.index_select(0, us), h.index_select(0, vs)])
        h = self.mlp(h).squeeze()
        E = h.new_zeros((n_nodes, n_nodes))
        E[tuple(edge_index)] += h
        return E



class InvariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='max'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim

        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 2, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, pos, edge_index):
        out = self.propagate(edge_index, pos=pos, h=h)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, index):
        t = scatter(pos_j, index, dim=0, reduce='mean')
        centroids = torch.index_select(t, 0, index)
        dist1 = torch.linalg.norm(pos_i - pos_j, axis=1)[:, None]
        dist2 = torch.linalg.norm(pos_j - centroids, axis=1)[:, None]

        msg = torch.cat([h_i, h_j, dist1, dist2], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):

        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class InvariantMPNNModel(Module):
    def __init__(self, num_layers=5, emb_dim=64, in_dim=4):
        super().__init__()

        self.lin_in = Linear(in_dim, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(InvariantMPNNLayer(emb_dim, aggr='max'))
        self.mlp = torch.nn.Sequential(
            Linear(64 * 2, 32),
            ReLU(),
            Linear(32, 1)
        )
        self.bn = BatchNorm1d(emb_dim)

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]
        pos = x[:, :2]
        h = self.lin_in(x[:, 2:])  # (n, d_n) -> (n, d)

        for conv in self.convs:
            h = self.bn(h)
            h = h + conv(h, pos, edge_index)  # (n, d) -> (n, d)

        us, vs = edge_index
        h = torch.column_stack([h.index_select(0, us), h.index_select(0, vs)])
        h = self.mlp(h).squeeze()

        E = h.new_zeros((n_nodes, n_nodes))
        E[tuple(edge_index)] += h
        return E


class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='max'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 2, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_pos = Sequential(
            Linear(2 * emb_dim + 2, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, 1)
        )

    def forward(self, h, pos, edge_index):
        out = self.propagate(edge_index, pos=pos, h=h)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, index):
        t = scatter(pos_j, index, dim=0, reduce='mean')
        centroids = torch.index_select(t, 0, index)
        dist1 = torch.linalg.norm(pos_i - pos_j, axis=1)[:, None]
        dist2 = torch.linalg.norm(pos_j - centroids, axis=1)[:, None]

        msg_h = torch.cat([h_i, h_j, dist1, dist2], dim=-1)
        msg_pos = torch.cat([h_i, h_j, dist1, dist2], dim=-1)

        return self.mlp_msg(msg_h), self.mlp_pos(msg_pos) * (pos_j - pos_i)

    def aggregate(self, inputs, index):
        aggr1 = scatter(inputs[0], index, dim=self.node_dim, reduce=self.aggr)
        aggr2 = scatter(inputs[1], index, dim=self.node_dim, reduce='sum')
        return aggr1, aggr2

    def update(self, aggr_out, h, pos):
        upd_out = torch.cat([h, aggr_out[0]], dim=-1)
        return self.mlp_upd(upd_out), pos + aggr_out[1]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class FinalMPNNModel(Module):
    def __init__(self, num_layers=5, emb_dim=64, in_dim=4):
        super().__init__()

        self.lin_in = Linear(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EquivariantMPNNLayer(emb_dim, aggr='max'))
        self.mlp = torch.nn.Sequential(
            Linear(emb_dim * 2, 32),
            ReLU(),
            Linear(32, 1)
        )
        self.bn = BatchNorm1d(emb_dim)

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]
        pos = x[:, :2]
        h = self.lin_in(x[:, 2:])

        for conv in self.convs:
            h = self.bn(h)
            h_update, pos_update = conv(h, pos, edge_index)
            h = h + h_update  # (n, d) -> (n, d)
            pos = pos_update  # (n, 3) -> (n, 3)

        us, vs = edge_index
        h = torch.column_stack([h.index_select(0, us), h.index_select(0, vs)])
        h = self.mlp(h).squeeze()

        E = h.new_zeros((n_nodes, n_nodes))
        E[tuple(edge_index)] += h
        return E
