import random
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, Linear, GCNConv
from torch.nn import ReLU, Softmax
from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from torch_geometric.utils.undirected import to_undirected
import collections
import time
import heapq
import pickle


TAU = 50
N_SAMPLES = 300
K = 5
BATCH_SIZE = 32
N_OBSTACLES = 6

class Model(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.lin1 = Linear(in_dim, 32)
        self.conv1 = GATConv(32, 64, edge_dim=10)
        self.conv2 = GATConv(64, 64, edge_dim=10)
        self.conv3 = GATConv(64, 64, edge_dim=10)
        self.conv4 = GATConv(64, 64, edge_dim=10)
        self.conv5 = GATConv(64, 64, edge_dim=10)
        self.sigmoid = torch.nn.Sigmoid()

        self.mlp = torch.nn.Sequential(
            Linear(64 * 2, 32),
            ReLU(),
            Linear(32, 1)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]

        x = self.lin1(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.conv5(x, edge_index)
        x = self.sigmoid(x)

        us, vs = edge_index
        x = torch.column_stack([x.index_select(0, us), x.index_select(0, vs)])
        x = self.mlp(x).squeeze()

        E = x.new_zeros((n_nodes, n_nodes))
        E[tuple(edge_index)] += x
        return E


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

    def not_collide(self, pos1, pos2):
        return not self.intersects(pos1.cpu().numpy(), pos2.cpu().numpy())

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

            if not (x1 >= ox0 and x0 <= ox1 or ox1 >= x0 and ox0 <= x1):
                continue

            p = m * ox0 + c
            q = m * ox1 + c

            if oy0 <= p <= oy1 or oy0 <= q <= oy1:
                #print(f'INTERSECTION {(x0, y0)} {(x1, y1)} {(obs_x, obs_y)}')
                return True

            if p < oy0 and oy1 < q:
                #print(f'INTERSECTION {(x0, y0)} {(x1, y1)} {(obs_x, obs_y)}')
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

    def visualise(self, graph, special_edges=torch.LongTensor([]), frontier_edges=torch.LongTensor([]), oracle_edge=None, opt_path=None):
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

        if opt_path:
            path_lines = []
            for u, v in opt_path:
                path_lines.append((pos[u], pos[v]))
            lc2 = matplotlib.collections.LineCollection(path_lines, colors='blue', linewidths=6)
            ax.add_collection(lc2)

        lines = [(pos[u.item()], pos[v.item()]) for [u, v] in special_edges.T]
        lc = matplotlib.collections.LineCollection(lines, colors='magenta', linewidths=3)
        ax.add_collection(lc)

        lines = [(pos[u.item()], pos[v.item()]) for [u, v] in frontier_edges.T]
        lc = matplotlib.collections.LineCollection(lines, colors='cyan', linewidths=3)
        ax.add_collection(lc)

        if oracle_edge is not None:
            lines = [(pos[oracle_edge[0].item()], pos[oracle_edge[1].item()])]
            lc = matplotlib.collections.LineCollection(lines, colors='yellow', linewidths=6)
            ax.add_collection(lc)

        ax.autoscale()
        ax.margins(0.1)
        plt.show()


class Scatter2D(Env2D):
    def __init__(self, *args):
        super().__init__(*args)

    def gen(self):
        for _ in range(1):
            x = 5 #random.randint(0, self.width - 1)
            y = 2 #random.randint(0, self.height - 1)
            for y_ in range(self.height):
                if y_ not in [y - 1, y, y + 1]:
                    self.obstacles.add((x, y_))



def dijkstra(pos, adj_list, src, dst):
    nodes = list(adj_list.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}

    dist[dst] = 0

    while nodes:
        cur = min(nodes, key=lambda node: dist[node])
        nodes.remove(cur)
        if dist[cur] == float('inf'):
            break

        for neighbor in adj_list[cur]:
            cost_new = dist[cur] + (pos[neighbor] - pos[cur]).pow(2).sum().pow(0.5).item()
            if cost_new < dist[neighbor]:
                dist[neighbor] = cost_new
                prev[neighbor] = cur

    path = deque()
    cur = src
    while prev[cur] is not None:
        path.append((cur, prev[cur]))
        cur = prev[cur]
    return list(path), dist, prev


def helper(E, env, graph, explore_steps, train_mode=False):
    tree_nodes = []
    tree_edges = []

    start_node = graph.x.shape[0] - 2
    goal_node = graph.x.shape[0] - 1

    tree_nodes.append(start_node)

    success = False

    steps = 0

    for steps in range(explore_steps):
        tree_nodes_ = torch.LongTensor(tree_nodes)

        us, vs = torch.where(E[tree_nodes_, :] != 0)

        if us.shape[0] == 0:
            break

        top = E[tree_nodes_[us], vs].argmax()

        start, end = tree_nodes[us[top].item()], vs[top].item()

        if train_mode or env.not_collide(graph.x[start, :2], graph.x[end, :2]):
            E[:, end] = 0
            tree_nodes.append(end)
            tree_edges.append([start, end])

            if end == goal_node:
                success = True
                break
        else:
            E[start, end] = 0
            E[end, start] = 0


        # adj_list = defaultdict(list)
        # for [u, v] in graph.edge_index.T:
        #     adj_list[u.item()].append(v.item())
        # start_node = graph.x.shape[0] - 2
        # goal_node = graph.x.shape[0] - 1
        # opt_path, dist, prev = dijkstra(graph.x[:, :2], adj_list, start_node, goal_node)
        # oracle_node = min(tree_nodes, key=lambda x: dist[x])
        # oracle_node_next = prev[oracle_node]
        # env.visualise(graph, special_edges=torch.LongTensor(tree_edges).T, oracle_edge=torch.LongTensor([oracle_node, oracle_node_next]), opt_path=opt_path)


    tree_nodes_ = torch.LongTensor(tree_nodes)
    us, vs = torch.where(E[tree_nodes_, :] != 0)
    frontier = torch.row_stack((tree_nodes_[us], vs))

    tree_edges = torch.LongTensor(tree_edges).T

    return tree_nodes, tree_edges, frontier, success, steps


def helper_(model, env, graph, explore_steps):
    start_node = graph.x.shape[0] - 2
    goal_node = graph.x.shape[0] - 1
    frontier = []
    frontier_set = set()

    def is_collide(edge):
        u, v = edge
        x0, y0 = graph.x[u][0].item(), graph.x[u][1].item()
        x1, y1 = graph.x[v][0].item(), graph.x[v][1].item()
        return env.intersects([x0, y0], [x1, y1])

    adj_list = defaultdict(list)
    for [u, v] in graph.edge_index.T:
        adj_list[u.item()].append(v.item())

    edge_priority = model(graph.x, graph.edge_index).squeeze()
    edge_to_index = {(edge[0].item(), edge[1].item()) : idx for idx, edge in enumerate(graph.edge_index.T)}

    for node in adj_list[start_node]:
        edge = (start_node, node)
        if not is_collide(edge):
            frontier.append((-edge_priority[edge_to_index[(start_node, node)]], edge))
        frontier_set.add(edge)

    heapq.heapify(frontier)

    tree_nodes = {start_node}
    tree_edges = set()

    for _ in range(explore_steps):
        # expand tree using highest priority edge

        if frontier == []:
            break

        (val, (u, v)) = heapq.heappop(frontier)

        tree_nodes.add(v)
        tree_edges.add((u, v))
        tree_edges.add((v, u))

        for node in adj_list[v]:
            new_edge = (v, node)
            assert v != node
            if new_edge not in frontier_set and (node, v) not in frontier_set and new_edge not in tree_edges:
                if not is_collide(new_edge):
                    new_item = (-edge_priority[edge_to_index[new_edge]], new_edge)
                    heapq.heappush(frontier, new_item)
                frontier_set.add(new_edge)

        frontier_set.remove((u, v))

        if v == goal_node:
            break

        #env.visualise(graph, special_edges=tree_edges, frontier_edges=list(frontier_set))

    return adj_list, start_node, goal_node, edge_priority, edge_to_index, tree_nodes, tree_edges, list(frontier_set)


def test():
    device = torch.device('cpu')
    model = Model(in_dim=6).to(device)
    model.load_state_dict(torch.load('models/cool.pth'))

    env = Scatter2D(10, 10, [0, 0], [10, 10])
    rg = env.rgg()

    adj_list = defaultdict(list)
    for [u, v] in rg.edge_index.T:
        adj_list[u.item()].append(v.item())
    start_node = rg.x.shape[0] - 2
    goal_node = rg.x.shape[0] - 1

    E = model(rg.x, rg.edge_index).squeeze()  # [n_nodes, n_nodes]

    tree_nodes, tree_edges, frontier, success, steps = helper(E, env, rg, 1231)

    opt_path, dist, prev = dijkstra(rg.x[:, :2], adj_list, start_node, goal_node)
    oracle_node = min(tree_nodes, key=lambda x: dist[x])
    oracle_node_next = prev[oracle_node]

    oracle_edge = None

    if oracle_node_next:
        oracle_edge = torch.LongTensor([oracle_node, oracle_node_next])

    env.visualise(rg, special_edges=tree_edges, frontier_edges=frontier, oracle_edge=oracle_edge, opt_path=opt_path)

    print(f'Success: {success} Steps: {steps}')


class CustomDataset:
    def __init__(self):
        self.instances = []

    def add_example(self, graph, adj_list, opt_path, dist, prev):
        self.instances.append((graph, adj_list, opt_path, dist, prev))

    def get(self, i):
        return self.instances[i]

    def len(self):
        return len(self.instances)

def make_data():
    dataset = CustomDataset()
    pbar = tqdm(range(1000))
    env = Scatter2D(10, 10, [0, 0], [10, 10])

    for _ in pbar:
        graph = env.rgg()

        adj_list = defaultdict(list)
        for [u, v] in graph.edge_index.T:
            adj_list[u.item()].append(v.item())
        start_node = graph.x.shape[0] - 2
        goal_node = graph.x.shape[0] - 1
        opt_path, dist, prev = dijkstra(graph.x[:, :2], adj_list, start_node, goal_node)

        dataset.add_example(graph, adj_list, opt_path, dist, prev)

        pbar.set_postfix_str(f'Generating RGGs')

    file = open('objs/train.pkl', 'wb')
    pickle.dump(dataset, file)


def train():
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(in_dim=6)#.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    env = Scatter2D(10, 10, [0, 0], [10, 10])

    file = open('objs/train.pkl', 'rb')
    dataset = pickle.load(file)

    idxs = list(np.arange(dataset.len()))

    for epoch in range(200):
        random.shuffle(idxs)

        pbar = tqdm(range(int(math.ceil(len(idxs) / BATCH_SIZE))))
        train_loss = 0

        for batch_idx in pbar:
            minibatch = idxs[batch_idx * BATCH_SIZE : min((batch_idx + 1) * BATCH_SIZE, len(idxs))]
            minibatch_loss = torch.tensor(0.)#.to(device)

            batch_size = 0

            for instance in minibatch:
                graph, adj_list, opt_path, dist, prev = dataset.get(instance)

                #start_time = time.process_time()
                E = model(graph.x, graph.edge_index).squeeze()  # [n_nodes, n_nodes]

                _, _, _, success, steps = helper(torch.clone(E), env, graph, explore_steps=1000)
                if not success:
                    continue
                tree_nodes, tree_edges, frontier, _, _ = helper(torch.clone(E), env, graph, explore_steps=random.randint(1, steps - 1))

                oracle_node = min(tree_nodes, key=lambda x : dist[x])
                oracle_node_next = prev[oracle_node]

                target_edge_idx = ((frontier.T == torch.LongTensor([oracle_node, oracle_node_next])).sum(1) == 2).nonzero().squeeze(1).item()
                minibatch_loss += - (E[tuple(frontier)].exp()[target_edge_idx] / E[tuple(frontier)].exp().sum()).log()
                batch_size += 1
                #print(f'b {time.process_time() - start_time}')

                #env.visualise(graph, special_edges=tree_edges, opt_path=opt_path)

                # tau_ = TAU
                # while True:
                #     #start = time.process_time()
                #
                #     explore_steps = random.randint(0, tau_)
                #     adj_list, start_node, goal_node, edge_priority, edge_to_index, tree_nodes, tree_edges, frontier = helper(model, env, graph, explore_steps)
                #
                #     #print(f'a {time.process_time() - start}')
                #
                #     if len(tree_edges) != 2 * explore_steps:
                #         #opt_path = dijkstra(graph.x[:, :2], adj_list, start_node, goal_node)
                #         #env.visualise(graph, special_edges=tree_edges, opt_path=opt_path)
                #         tau_ //= 2
                #         continue
                #
                #     #start = time.process_time()
                #
                #     opt_path = dijkstra(graph.x[:, :2], adj_list, start_node, goal_node)
                #
                #     #print(f'b {time.process_time() - start}')
                #
                #     break
                #
                # #start = time.process_time()
                #
                # frontier_edges = [edge_to_index[edge] for edge in frontier]
                # t = edge_priority.index_select(0, torch.LongTensor(frontier_edges).to(device))
                # p = None
                #
                # if idx == BATCH_SIZE - 1:
                #     env.visualise(graph, special_edges=tree_edges, opt_path=opt_path)
                #
                # for u, v in opt_path:
                #     if u in tree_nodes and v not in tree_nodes:
                #         p = edge_priority[edge_to_index[(u, v)]]
                #         break
                #
                # if p is None:
                #     continue
                #
                # minibatch_loss += -torch.log(p.exp() / t.exp().sum())

                #print(f'c {time.process_time() - start}')

            minibatch_loss /= batch_size

            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()

            train_loss += minibatch_loss
            pbar.set_postfix_str(f'Batch loss {minibatch_loss:.5f} Total train loss {train_loss:.5f}')
            torch.save(model.state_dict(), 'models/cool2.pth')


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 120
    train()

    #

