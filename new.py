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
import torch_scatter

TAU = 50
N_SAMPLES = 200
K = 10
BATCH_SIZE = 16
REPLAY_SIZE = 1024
N_OBSTACLES = 6

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, width, height, start=None, end=None, force_dist=0):
        self.width = width
        self.height = height
        self.obstacles = set()

        self.k = K
        self.n_samples = N_SAMPLES

        self.start = start
        self.end = end
        self.force_dist = force_dist

        self.gen()
        self.set_start_end()

    def gen(self):
        raise NotImplementedError()

    def not_in_obstacle(self, pos):
        for obs_x, obs_y in self.obstacles:
            ox0 = obs_x
            ox1 = obs_x + 1
            oy0 = obs_y
            oy1 = obs_y + 1
            if ox0 <= pos[0] <= ox1 and oy0 <= pos[1] <= oy1:
                return False
        return True

    def not_collide(self, pos1, pos2):
        return not self.intersects(pos1.cpu().numpy(), pos2.cpu().numpy())

    def intersects(self, pos1, pos2):
        x0, y0 = pos1
        x1, y1 = pos2

        if abs(x1 - x0) < 10e-5:
            for obs_x, obs_y in self.obstacles:
                oy0 = obs_y
                oy1 = obs_y + 1
                if y1 >= oy0 and y0 <= oy1 or oy1 >= y0 and oy0 <= y1:
                    return True
            return False

        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

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
                return True

            if p < oy0 and oy1 < q or p > oy0 and oy1 > q:
                return True

        return False

    def graph_properties(self, graph):
        adj_list = defaultdict(list)
        for [u, v] in graph.edge_index.T:
            adj_list[u.item()].append(v.item())
        start_node = graph.x.shape[0] - 2
        goal_node = graph.x.shape[0] - 1

        costs = {}
        pos = graph.x[:, :2]
        for node in range(graph.x.shape[0]):
            for nb in adj_list[node]:
                if self.not_collide(pos[nb], pos[node]):
                    dist = (pos[nb] - pos[node]).pow(2).sum().pow(0.5).item()
                else:
                    dist = float('inf')
                costs[(node, nb)] = dist

        opt_path, dist, prev = dijkstra(adj_list, costs, start_node, goal_node)

        return graph, adj_list, costs, opt_path, dist, prev

    def set_start_end(self):
        while True:
            self.start = torch.rand(2) * torch.tensor([self.width, self.height])
            self.end = torch.rand(2) * torch.tensor([self.width, self.height])
            if (self.start - self.end).pow(2).sum().pow(0.5) < self.force_dist:
                continue
            if self.not_in_obstacle(self.start) and self.not_in_obstacle(self.end):
                break

    def start_end_seeded(self, n, s):
        torch.random.manual_seed(s)
        ret = []
        for _ in range(n):
            while True:
                start = torch.rand(2) * torch.tensor([self.width, self.height])
                end = torch.rand(2) * torch.tensor([self.width, self.height])
                if self.not_in_obstacle(start) and self.not_in_obstacle(end):
                    break
            ret.append([
                list(start.numpy()),
                list(end.numpy())
            ])
        return ret

    def rand_position(self):
        return torch.rand(2) * torch.tensor([self.width, self.height])

    def rgg(self, force_path=False):
        while True:
            graph = self._rgg()
            graph, adj_list, costs, opt_path, dist, prev = self.graph_properties(graph)
            if opt_path == [] and force_path:
                continue
            return self, graph, adj_list, costs, opt_path, dist, prev

    def _rgg(self):
        rgg_size = self.n_samples + 2

        r = torch.rand((self.n_samples, 2)) * torch.tensor([self.width, self.height])  # [[x1,y1],[x2,y2],...,[xn,yn]]

        r = torch.row_stack([r, self.start, self.end])

        l2 = (r - self.end).pow(2).sum(dim=1)
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

        D = (r[:, 0][:, None] - r[:, 0].repeat(rgg_size, 1)).pow(2) + (
                    r[:, 1][:, None] - r[:, 1].repeat(rgg_size, 1)).pow(2)
        D += torch.eye(rgg_size) * (2 ** 20)
        knn = D.argsort(dim=1)[:, :self.k]

        us = torch.arange(rgg_size).repeat(1, self.k).squeeze()
        vs = knn.T.reshape(-1, 1).squeeze()

        edge_index = to_undirected(torch.row_stack([us, vs]))
        return Data(x=r, edge_index=edge_index)

    def visualise(self, graph, special_edges=torch.LongTensor([]), frontier_edges=torch.LongTensor([]),
                  oracle_edge=None, opt_path=None):
        pos = graph.x[:, :2].detach().cpu().numpy()
        px_free = [x for idx, [x, y] in enumerate(pos) if graph.x[idx, 3] == 1]
        py_free = [y for idx, [x, y] in enumerate(pos) if graph.x[idx, 3] == 1]

        px_collide = [x for idx, [x, y] in enumerate(pos) if graph.x[idx, 4] == 1]
        py_collide = [y for idx, [x, y] in enumerate(pos) if graph.x[idx, 4] == 1]

        fig, ax = plt.subplots()

        for x, y in self.obstacles:
            rect = plt.Rectangle((x, y), 1, 1, color='purple')
            ax.add_artist(rect)

        ax.scatter(px_free, py_free, c='lime', alpha=0.5)
        ax.scatter(px_collide, py_collide, c='red')

        us, vs = graph.edge_index
        lines = [(pos[us[i].item()], pos[vs[i].item()]) for i in range(graph.edge_index.shape[1])]
        lc = matplotlib.collections.LineCollection(lines, colors='green', linewidths=2, alpha=0.2)
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
        lc = matplotlib.collections.LineCollection(lines, colors='cyan', linewidths=3, alpha=0.5)
        ax.add_collection(lc)

        if oracle_edge is not None:
            lines = [(pos[oracle_edge[0].item()], pos[oracle_edge[1].item()])]
            lc = matplotlib.collections.LineCollection(lines, colors='yellow', linewidths=6)
            ax.add_collection(lc)

        ax.scatter([pos[-2, 0]], [pos[-2, 1]], marker='s', s=120, c='gold', zorder=2)
        ax.scatter([pos[-1, 0]], [pos[-1, 1]], marker='*', s=250, c='gold', zorder=2)

        ax.autoscale()
        ax.margins(0.1)
        plt.show()


class Scatter2D(Env2D):
    def __init__(self, *args, map=None, p=0.1, **kwargs):
        self.p = p

        super().__init__(*args, **kwargs)
        if map is not None:
            self.obstacles = set()
            map = torch.tensor(map).reshape((self.height, self.width)).flip(0)
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] == 1:
                        self.obstacles.add((j, i))

    def gen(self):
        while True:
            a = torch.rand((self.height, self.width))
            a = (a < self.p).long()
            num_free = np.prod(a.shape) - a.sum()
            if num_free == 0:
                continue
            b = torch.nn.functional.pad(a, [1, 1, 1, 1], value=1)
            i = j = 0
            for i in range(b.shape[0]):
                f = False
                for j in range(b.shape[1]):
                    if b[i, j] == 0:
                        f = True
                        break
                if f:
                    break
            count = 0
            stack = [(i, j)]
            b[i, j] = 1

            while stack != []:
                i, j = stack.pop()
                count += 1
                if b[i - 1, j] == 0:
                    stack.append((i - 1, j))
                    b[i - 1, j] = 1
                if b[i + 1, j] == 0:
                    stack.append((i + 1, j))
                    b[i + 1, j] = 1
                if b[i, j - 1] == 0:
                    stack.append((i, j - 1))
                    b[i, j - 1] = 1
                if b[i, j + 1] == 0:
                    stack.append((i, j + 1))
                    b[i, j + 1] = 1

            if count == num_free:
                for x in range(self.width):
                    for y in range(self.height):
                        if a[y, x] == 1:
                            self.obstacles.add((x, y))
                break



def dijkstra(adj_list, costs, src, dst):
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
            cost_new = dist[cur] + costs[(cur, neighbor)]
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
    E[:, start_node] = 0

    success = False

    steps = 0

    for steps in range(1, explore_steps + 1):
        tree_nodes_ = torch.LongTensor(tree_nodes)

        us, vs = torch.where(E[tree_nodes_, :] != 0)

        if us.shape[0] == 0:
            break

        top = E[tree_nodes_[us], vs].argmax()

        start, end = tree_nodes[us[top].item()], vs[top].item()

        # TODO: CHECK
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


def tree_to_path(adj_list, nodes, pos, src, dst):
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}

    dist[dst] = 0

    while nodes:
        cur = min(nodes, key=lambda node: dist[node])
        nodes.remove(cur)
        if dist[cur] == float('inf'):
            break

        for neighbor in adj_list[cur]:
            cost_new = dist[cur] + (pos[cur] - pos[neighbor]).pow(2).sum().pow(0.5).item()
            if cost_new < dist[neighbor]:
                dist[neighbor] = cost_new
                prev[neighbor] = cur

    path = deque()
    cur = src
    while prev[cur] is not None:
        path.append((cur, prev[cur]))
        cur = prev[cur]
    return list(path), dist[src]


def evaluate(evaluation_dataset, model):
    envs = evaluation_dataset.envs

    success_count = 0
    running_time_total = 0
    path_cost_total = 0
    n_instances = len(envs)

    for env_instance in tqdm(envs):
        time_start = time.time()

        success = False
        tries = 0

        while not success and tries < 3:
            graph = env_instance._rgg()
            E = model(graph.x, graph.edge_index).squeeze()  # [n_nodes, n_nodes]
            tree_nodes, tree_edges, frontier, success, steps = helper(E, env_instance, graph, 1000)
            tries += 1

        if not success:
            continue

        #env.visualise(graph, special_edges=tree_edges, frontier_edges=frontier)

        adj_list = defaultdict(list)
        nodes = set()
        for [u, v] in tree_edges.T:
            adj_list[v.item()].append(u.item())  # need to revert edge directions in tree
            nodes |= {u.item(), v.item()}

        path, distance = tree_to_path(adj_list, nodes, pos=graph.x[:, :2], src=graph.x.shape[0] - 2, dst=graph.x.shape[0] - 1)
        assert path != []
        success_count += 1
        path_cost_total += distance
        running_time_total += time.time() - time_start

    print(f'Success rate: {success_count / n_instances : .3f}')
    print(f'Average time: {running_time_total / success_count : .3f}')
    print(f'Average cost: {path_cost_total / success_count : .3f}')


def evaluate_baseline(evaluation_dataset):
    envs = evaluation_dataset.envs

    success_count = 0
    running_time_total = 0
    path_cost_total = 0
    n_instances = len(envs)

    for env_instance in tqdm(envs):
        time_start = time.time()

        _, graph, adj_list, costs, opt_path, dist, prev = env_instance.rgg(force_path=True)

        if opt_path != []:
            success_count += 1
        
        path_cost_total += dist[graph.x.shape[0] - 2]
        running_time_total += time.time() - time_start

    print(f'Success rate: {success_count / n_instances : .3f}')
    print(f'Average time: {running_time_total / success_count : .3f}')
    print(f'Average cost: {path_cost_total / success_count : .3f}')


def evaluate_rrt(evaluation_dataset):
    envs = evaluation_dataset.envs

    success_count = 0
    running_time_total = 0
    path_cost_total = 0
    n_instances = len(envs)

    max_iters = 1000
    eps = 0.3

    for env_instance in tqdm(envs):
        time_start = time.time()

        start_pos = np.array(env_instance.start)
        end_pos = np.array(env_instance.end)

        adj_list = defaultdict(list)
        positions = [start_pos]
        nodes = [0]

        success = False

        for _ in range(max_iters):
            rand_pos = env_instance.rand_position().numpy()
            nearest = min(nodes, key=lambda n: np.linalg.norm(positions[n] - rand_pos))
            vec = rand_pos - positions[nearest]
            norm = np.linalg.norm(vec)
            unit = vec / norm
            delta = min(eps, norm)
            new_node_pos = positions[nearest] + delta * unit

            if env_instance.intersects(new_node_pos, positions[nearest]):
                continue

            new_node = nodes[-1] + 1
            nodes.append(new_node)
            positions.append(new_node_pos)

            adj_list[new_node].append(nearest)

            if np.linalg.norm(new_node_pos - end_pos) < eps:
                new_node = nodes[-1] + 1
                nodes.append(new_node)
                positions.append(end_pos)
                adj_list[new_node].append(new_node - 1)
                success = True

            if success:
                break

        if not success:
            continue

        path, distance = tree_to_path(adj_list, set(nodes), pos=torch.tensor(np.array(positions)), src=0, dst=new_node)
        assert path != []
        success_count += 1
        path_cost_total += distance
        running_time_total += time.time() - time_start

    print(f'Success rate: {success_count / n_instances : .3f}')
    print(f'Average time: {running_time_total / success_count : .3f}')
    print(f'Average cost: {path_cost_total / success_count : .3f}')


def test(model, env):
    env, graph, adj_list, costs, opt_path, dist, prev = env.rgg(force_path=True)

    print('test')
    E = model(graph.x, graph.edge_index).squeeze()  # [n_nodes, n_nodes]
    tree_nodes, tree_edges, frontier, success, steps = helper(E, env, graph, 1000)

    oracle_node = min(tree_nodes, key=lambda x: dist[x])
    oracle_node_next = prev[oracle_node]

    oracle_edge = None

    if oracle_node_next:
        oracle_edge = torch.LongTensor([oracle_node, oracle_node_next])

    env.visualise(graph, special_edges=tree_edges, frontier_edges=frontier, oracle_edge=oracle_edge, opt_path=opt_path)

    print(f'Success: {success} Steps: {steps}')


class CustomDataset:
    def __init__(self):
        self.instances = []
        self.env_mapping = {}
        self.envs = []

    def add_example(self, env, graph, adj_list, costs, opt_path, dist, prev):
        self.instances.append((graph, adj_list, costs, opt_path, dist, prev))
        if env not in self.envs:
            self.envs.append(env)
        self.env_mapping[self.len() - 1] = self.envs.index(env)

    def get(self, i):
        return self.instances[i]

    def len(self):
        return len(self.instances)

    def get_env(self, i):
        return self.envs[self.env_mapping[i]]


class EvaluationDataset:
    def __init__(self, envs):
        self.envs = envs


def make_data(name, envs):
    dataset = CustomDataset()
    pbar = tqdm(range(len(envs)))

    for idx in pbar:
        dataset.add_example(*envs[idx].rgg(force_path=True))
        pbar.set_postfix_str(f'Generating RGGs')

    file = open(f'objs/{name}.pkl', 'wb')
    pickle.dump(dataset, file)


def create_mini_batch(graph_list, frontiers, target_edges):
    batch_edge_index = graph_list[0].edge_index
    batch_x = graph_list[0].x
    batch_frontiers = frontiers[0]
    batch_batch = torch.zeros((graph_list[0].x.shape[0]), dtype=torch.int64)
    batch_frontier_batch = torch.zeros((frontiers[0].shape[1]), dtype=torch.int64)
    batch_target_edges = torch.LongTensor(target_edges[0]).unsqueeze(1)

    for idx, graph in enumerate(graph_list[1:]):
        batch_x = torch.row_stack([batch_x, graph.x])
        batch_frontiers = torch.column_stack((batch_frontiers, frontiers[1 + idx] + batch_batch.shape[0]))

        batch_edge_index = torch.column_stack([batch_edge_index, graph.edge_index + batch_batch.shape[0]])

        batch_target_edges = torch.column_stack([
                batch_target_edges,
                batch_batch.shape[0] + torch.LongTensor(target_edges[1 + idx]).unsqueeze(1)
            ])

        batch_batch = torch.cat([batch_batch, 1 + idx + torch.zeros((graph.x.shape[0]), dtype=torch.int64)])
        batch_frontier_batch = torch.cat([batch_frontier_batch, 1 + idx + torch.zeros((frontiers[1 + idx].shape[1]), dtype=torch.int64)])

    batch_graph = Data(x=batch_x, edge_index=batch_edge_index)
    return batch_graph, batch_frontiers, batch_target_edges, batch_frontier_batch



def train(train_path, model_name, epochs=100, fast=False):
    model = Model(in_dim=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    file = open(train_path, 'rb')
    dataset = pickle.load(file)
    file.close()

    idxs = list(np.arange(dataset.len()))

    replay_buffer = [None] * REPLAY_SIZE
    replay_pointer = 0

    model.train()

    for epoch in range(epochs):
        random.shuffle(idxs)

        train_loss = 0
        best_loss = float('inf')

        if fast:
            pbar = tqdm(idxs)

            for idx in pbar:
                graph, adj_list, costs, opt_path, dist, prev = dataset.get(idx)
                env = dataset.get_env(idx)

                with torch.no_grad():
                    graph = Data(graph.x.to(device), graph.edge_index.to(device))
                    E = model(graph.x, graph.edge_index).squeeze().cpu()  # [n_nodes, n_nodes]

                    _, _, _, success, steps = helper(torch.clone(E), env, graph, explore_steps=1000000)
                    assert success
                    tree_nodes, tree_edges, frontier, _, _ = helper(torch.clone(E), env, graph,
                                                                    explore_steps=random.randint(0, steps - 1))

                    frontier = frontier.to(device)

                    # Compute the best next edge to add to the tree according to the oracle
                    oracle_node = min(tree_nodes, key=lambda x: dist[x])
                    oracle_node_next = prev[oracle_node]

                    #target_edge_idx = ((frontier.T == torch.LongTensor([oracle_node, oracle_node_next]).to(device)).sum(
                    #    1) == 2).nonzero().squeeze(1).item()

                    replay_buffer[replay_pointer % REPLAY_SIZE] = (graph, frontier, [oracle_node, oracle_node_next])
                    replay_pointer += 1

                # sample batch from replay buffer and backprop
                if replay_pointer >= BATCH_SIZE:
                    batch_idxs = random.sample(list(np.arange(min(replay_pointer, REPLAY_SIZE))), BATCH_SIZE)
                    batch = [replay_buffer[i] for i in batch_idxs]
                    graph_list = []
                    frontiers = []
                    target_edges = []
                    for graph, frontier, target_edge in batch:
                        graph_list.append(graph)
                        frontiers.append(frontier)
                        target_edges.append(target_edge)
                    batch_graph, batch_frontier, batch_target_edges, batch_frontier_batch = create_mini_batch(graph_list, frontiers, target_edges)

                    batch_target_edges = batch_target_edges.to(device)
                    batch_frontier_batch = batch_frontier_batch.to(device)

                    E = model(batch_graph.x, batch_graph.edge_index).squeeze()

                    frontier_vals = E[tuple(batch_frontier)]
                    target_vals = E[tuple(batch_target_edges)]

                    t1 = target_vals.exp()
                    t2 = torch_scatter.scatter_add(frontier_vals.exp(), index=batch_frontier_batch)

                    minibatch_loss = -(t1 / t2).log().mean()

                    optimizer.zero_grad()
                    minibatch_loss.backward()
                    optimizer.step()

                    train_loss += minibatch_loss

                    pbar.set_postfix_str(f'Batch loss {minibatch_loss:.5f} Total train loss {train_loss:.5f}')

                    if train_loss < best_loss:
                        best_loss = train_loss
                        torch.save(model.state_dict(), f'models/{model_name}.pth')
        else:

            pbar = tqdm(range(int(math.ceil(len(idxs) / BATCH_SIZE))))

            for batch_idx in pbar:
                minibatch = idxs[batch_idx * BATCH_SIZE: min((batch_idx + 1) * BATCH_SIZE, len(idxs))]
                minibatch_loss = torch.tensor(0.) #.to(device)

                batch_size = 0

                for instance in minibatch:
                    graph, adj_list, costs, opt_path, dist, prev = dataset.get(instance)
                    env = dataset.get_env(instance)

                    # start_time = time.process_time()
                    E = model(graph.x.to(device), graph.edge_index.to(device)).squeeze().cpu() # [n_nodes, n_nodes]

                    _, _, _, success, steps = helper(torch.clone(E), env, graph, explore_steps=1000000)
                    assert success
                    tree_nodes, tree_edges, frontier, _, _ = helper(torch.clone(E), env, graph,
                                                                    explore_steps=random.randint(0, steps - 1))

                    # Compute the best next edge to add to the tree according to the oracle
                    oracle_node = min(tree_nodes, key=lambda x: dist[x])
                    oracle_node_next = prev[oracle_node]

                    target_edge_idx = ((frontier.T == torch.LongTensor([oracle_node, oracle_node_next])).sum(
                         1) == 2).nonzero().squeeze(1).item()
                    minibatch_loss += - (E[tuple(frontier)].exp()[target_edge_idx] / E[tuple(frontier)].exp().sum()).log()

                    #print(E[tuple(frontier)])

                    batch_size += 1

                minibatch_loss /= batch_size

                optimizer.zero_grad()
                minibatch_loss.backward()
                optimizer.step()

                train_loss += minibatch_loss
                pbar.set_postfix_str(f'Batch loss {minibatch_loss:.5f} Total train loss {train_loss:.5f}')

                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(model.state_dict(), f'models/{model_name}.pth')


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['figure.figsize'] = (6, 6)

    model = Model(in_dim=6)

    model1 = Model(in_dim=6)
    model1.load_state_dict(torch.load('models/fast-1-epoch.pth'))

    model2 = Model(in_dim=6)
    model2.load_state_dict(torch.load('models/fast-5-epoch.pth'))

    model3 = Model(in_dim=6)
    model3.load_state_dict(torch.load('models/fast-10-epoch.pth'))

    model4 = Model(in_dim=6)
    model4.load_state_dict(torch.load('models/fast-epochs=10-buffer=1024-batch=32.pth'))

    model_base = Model(in_dim=6)
    model_base.load_state_dict(torch.load('models/model_random.pth'))

    world_empty = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    world_basic = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ]

    world_checker = [
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    world_x = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    world_corridor = [
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    ]

    world_scatter = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ]

    # env_train = Scatter2D(10, 10, map=world_basic)
    # env_checker = Scatter2D(10, 10, map=world_checker)
    # env_corridor = Scatter2D(10, 10, map=world_corridor)
    # env_x = Scatter2D(10, 10, map=world_x)
    #env_scatter = Scatter2D(10, 10, map=world_scatter)

    #envs_train = [Scatter2D(10, 10, p=0.08) for _ in range(1000)]
    #make_data('scatter_random_n=200_k=10_s=1000', envs_train)

    if False:
        evaluation_dataset = EvaluationDataset([Scatter2D(10, 10, force_dist=0, p=0.08) for _ in range(1000)])

        #evaluate(evaluation_dataset, model)
        #evaluate(evaluation_dataset, model1)
        evaluate(evaluation_dataset, model3)
        evaluate(evaluation_dataset, model4)
        evaluate_rrt(evaluation_dataset)

    #train('objs/scatter_random_n=200_k=10_s=1000.pkl', 'slow', 30, fast=False)
    #train('objs/scatter_random_n=200_k=10_s=1000.pkl', 'fast-1-epoch', 1, fast=True)
    #train('objs/scatter_random_n=200_k=10_s=1000.pkl', 'fast-5-epoch', 5, fast=True)
    train('objs/scatter_random_n=200_k=10_s=1000.pkl', 'fast-epochs=50-buffer=1024-batch=64', 32, fast=True)


#%%
