import random
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, Linear
from torch.nn import ReLU, Softmax
from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from torch_geometric.utils.undirected import to_undirected

TAU = 10
N_SAMPLES = 200
K = 5

class Model(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.lin1 = Linear(in_dim, 32)
        self.conv1 = GATConv(32, 64, edge_dim=10)
        self.conv2 = GATConv(64, 64, edge_dim=10)
        self.conv3 = GATConv(64, 64, edge_dim=10)
        self.mlp = torch.nn.Sequential(
            Linear(64 * 2, 32),
            ReLU(),
            Linear(32, 1)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)

        us, vs = edge_index
        t = torch.column_stack([x.index_select(0, us), x.index_select(0, vs)])
        return self.mlp(t)


class Env2D:
    def __init__(self, width, height, start, end):
        self.width = width
        self.height = height
        self.start = start
        self.end = end
        self.obstacles = set()

        self.k = K
        self.n_samples = N_SAMPLES

        self.gen()

    def gen(self):
        raise NotImplementedError()

    def intersects(self, pos1, pos2):
        x0, y0 = pos1
        x1, y1 = pos2
        m = (y1 - y0) / (x1 - x0)
        c = y0 - m * x0

        for obs_x, obs_y in self.obstacles:
            ox0 = obs_x
            ox1 = obs_x + 1
            oy0 = obs_y
            oy1 = obs_y + 1

            p = m * ox0 + c
            q = m * ox1 + c

            if oy0 < p < oy1 or oy0 < q < oy1:
                return True

            if p < oy0 and oy1 < q:
                return True

        return False


    def rgg(self):
        rgg_size = self.n_samples + 2

        r = torch.rand((self.n_samples, 2)) * torch.tensor([self.width, self.height]) # [[x1,y1],[x2,y2],...,[xn,yn]]
        r = torch.row_stack([r, torch.tensor([self.start, self.end])])

        l2 = (r - torch.tensor(self.end)).pow(2).sum(dim=1)
        r = torch.column_stack([r, l2])

        # check if each point is in free space or obstacle or goal
        one_hot = []
        for idx, [x, y] in enumerate(r[:, :2].floor().long()):
            if idx == r.shape[0] - 1:
                one_hot.append([0, 0, 1])
            elif (x.item(), y.item()) in self.obstacles:
                one_hot.append([0, 1, 0])
            else:
                one_hot.append([1, 0, 0])
        r = torch.column_stack([r, torch.tensor(one_hot)])

        D = (r[:, 0][:, None] - r[:, 0].repeat(rgg_size, 1)).pow(2) + (r[:, 1][:, None] - r[:, 1].repeat(rgg_size, 1)).pow(2)
        D += torch.eye(rgg_size) * (2 ** 20)
        knn = D.argsort(dim=1)[:, :self.k]

        us = torch.arange(rgg_size).repeat(1, self.k).squeeze()
        vs = knn.T.reshape(-1, 1).squeeze()

        edge_index = to_undirected(torch.row_stack([us, vs]))
        return Data(x=r, edge_index=edge_index)

    def visualise(self, graph):
        pos = graph.x[:, :2].detach().cpu().numpy()
        px_free = [x for idx, [x, y] in enumerate(pos) if graph.x[idx, 3] == 1]
        py_free = [y for idx, [x, y] in enumerate(pos) if graph.x[idx, 3] == 1]

        px_collide = [x for idx, [x, y] in enumerate(pos) if graph.x[idx, 4] == 1]
        py_collide = [y for idx, [x, y] in enumerate(pos) if graph.x[idx, 4] == 1]


        fig, ax = plt.subplots()

        for x, y in self.obstacles:
            rect = plt.Rectangle((x, y), 1, 1, color='purple')
            ax.add_artist(rect)

        ax.scatter(px_free, py_free, c='cyan')
        ax.scatter(px_collide, py_collide, c='red')

        ax.scatter(graph.x[-2, 0], graph.x[-2, 1], c='black')
        ax.scatter(graph.x[-1, 0], graph.x[-1, 1], c='black')

        us, vs = graph.edge_index
        lines = [(pos[us[i].item()], pos[vs[i].item()]) for i in range(graph.edge_index.shape[1])]
        lc = matplotlib.collections.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)

        adj_list = defaultdict(list)
        for [u, v] in graph.edge_index.T:
            adj_list[u.item()].append(v.item())
        opt_path = dijkstra(adj_list, graph.x.shape[0] - 2, graph.x.shape[0] - 1)

        path_lines = []
        for u, v in opt_path:
            path_lines.append((pos[u], pos[v]))
        # if path is not None:
        #     paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = matplotlib.collections.LineCollection(path_lines, colors='blue', linewidths=3)
        ax.add_collection(lc2)

        ax.autoscale()
        ax.margins(0.1)
        plt.show()


class Scatter2D(Env2D):
    def __init__(self, *args):
        super().__init__(*args)

    def gen(self):
        for _ in range(0):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.obstacles.add((x, y))


def dijkstra(adj_list, src, dst):
    nodes = list(adj_list.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[src] = 0

    while nodes:
        cur = min(nodes, key=lambda node: dist[node])
        nodes.remove(cur)
        if dist[cur] == float('inf'):
            break

        for neighbor in adj_list[cur]:
            cost_new = dist[cur] + 1
            if cost_new < dist[neighbor]:
                dist[neighbor] = cost_new
                prev[neighbor] = cur

    path = deque()
    cur = dst
    while prev[cur] is not None:
        path.appendleft((prev[cur], cur))
        cur = prev[cur]
    return list(path)



def train():
    model = Model(in_dim=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    env = Scatter2D(10, 10, [0, 0], [10, 10])
    dataset = []

    pbar = tqdm(range(1000))
    for _ in pbar:
        dataset.append(env.rgg())
        pbar.set_postfix_str(f'Generating RGG')

    train_dataset = dataset[:500]
    test_dataset = dataset[500:]


    for epoch in range(20):
        pbar = tqdm(range(int(math.ceil(len(train_dataset) / 128))))
        train_loss = 0

        for batch_idx in pbar:
            minibatch = train_dataset[batch_idx * 128 : min((batch_idx + 1) * 128, len(train_dataset))]
            minibatch_loss = torch.tensor(0.)

            for graph in minibatch:
                start_node = graph.x.shape[0] - 2
                goal_node = graph.x.shape[0] - 1
                frontier = []

                adj_list = defaultdict(list)
                for [u, v] in graph.edge_index.T:
                    adj_list[u.item()].append(v.item())

                opt_path = dijkstra(adj_list, start_node, goal_node)

                for node in adj_list[start_node]:
                    frontier.append((start_node, node))

                edge_priority = model(graph.x, graph.edge_index).squeeze()
                edge_to_index = {(edge[0].item(), edge[1].item()) : idx for idx, edge in enumerate(graph.edge_index.T)}

                explore_steps = random.randint(0, TAU)
                tree_nodes = {start_node}

                # perform initial exploration for random number of steps
                for _ in range(explore_steps):
                    frontier_edges = [edge_to_index[edge] for edge in frontier]
                    frontier_priorities = edge_priority.index_select(0, torch.LongTensor(frontier_edges)).argsort().detach().cpu().numpy()
                    chosen = None
                    # expand tree using highest priority edge
                    for pos in frontier_priorities:
                        u, v = frontier[pos]
                        x0, y0 = graph.x[u][0].item(), graph.x[u][1].item()
                        x1, y1 = graph.x[v][0].item(), graph.x[v][1].item()

                        if not env.intersects([x0, y0], [x1, y1]):
                            chosen = frontier[pos]
                            break

                    if chosen is None:
                        break

                    frontier.remove(chosen)

                    tree_nodes.add(chosen[1])

                    for node in adj_list[chosen[1]]:
                        frontier.append((chosen[1], node))

                if len(tree_nodes) != 1 + explore_steps:
                    #print('SKIP')
                    continue

                frontier_edges = [edge_to_index[edge] for edge in frontier]
                t = edge_priority.index_select(0, torch.LongTensor(frontier_edges))
                p = None
                for u, v in opt_path:
                    if u in tree_nodes and v not in tree_nodes:
                        p = edge_priority[edge_to_index[(u, v)]]
                        break

                if p is None:
                    continue

                minibatch_loss += -torch.log(p.exp() / t.exp().sum())

            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()

            train_loss += minibatch_loss
            pbar.set_postfix_str(f'Batch loss {minibatch_loss:.5f} Total train loss {train_loss:.5f}')


if __name__ == '__main__':
    env = Scatter2D(10, 10, [0, 0], [10, 10])
    rg = env.rgg()
    env.visualise(rg)
    train()