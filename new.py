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
from envs import *

TAU = 50
N_SAMPLES = 100 #200
K = 7 #7
BATCH_SIZE = 32
REPLAY_SIZE = 1024
N_OBSTACLES = 6

TRAIN_SET = 'scatter_random_n=200_k=10_p=0.08_s=1000'


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


def helper(E, env, graph, explore_steps, train_mode=False, rnd=False):
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

        if not rnd:
            top = E[tree_nodes_[us], vs].argmax()
        else:
            top = random.randint(0, us.shape[0] - 1)

        start, end = tree_nodes[us[top].item()], vs[top].item()

        # TODO: CHECK
        if env.not_collide(graph.x[start, :env.n_degrees], graph.x[end, :env.n_degrees]):
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


def evaluate(env, env_data, model, rnd=False):
    success_count = 0
    running_time_total = 0
    path_cost_total = 0
    n_instances = len(env_data)

    for env_data_ in tqdm(env_data):
        time_start = time.time()

        success = False
        tries = 0

        env.load_data(env_data_)

        while not success and tries < 3:
            graph = env._rgg()
            E = model(graph.x, graph.edge_index).squeeze()  # [n_nodes, n_nodes]
            tree_nodes, tree_edges, frontier, success, steps = helper(E, env, graph, 1000, rnd=rnd)
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


def evaluate_rrt(env_instance, env_data):
    success_count = 0
    running_time_total = 0
    path_cost_total = 0
    n_instances = len(env_data)

    max_iters = 5000
    eps = 0.3

    pbar = tqdm(env_data)

    for env_data_ in pbar:
        env_instance.load_data(env_data_)

        time_start = time.time()

        start_pos = np.array(env_instance.start)
        end_pos = np.array(env_instance.end)

        adj_list = defaultdict(list)
        positions = [start_pos]
        nodes = [0]

        success = False

        closest = float('inf')
        closest_v = None

        for _ in range(max_iters):
            rand_pos = env_instance.rand_position().numpy()

            nearest = min(nodes, key=lambda n: np.linalg.norm(positions[n] - rand_pos))
            vec = rand_pos - positions[nearest]
            norm = np.linalg.norm(vec)
            unit = vec / norm
            delta = min(eps, norm)
            new_node_pos = positions[nearest] + delta * unit

            #if not env_instance.not_collide(torch.tensor(new_node_pos), torch.tensor(positions[nearest])):
            #    continue

            new_node = nodes[-1] + 1
            nodes.append(new_node)
            positions.append(new_node_pos)

            adj_list[new_node].append(nearest)

            d = np.linalg.norm(new_node_pos - end_pos)

            if d < closest:
                closest = d
                closest_v = new_node_pos

            if d < eps:
                new_node = nodes[-1] + 1
                nodes.append(new_node)
                positions.append(end_pos)
                adj_list[new_node].append(new_node - 1)
                success = True

            if success:
                break

        pbar.set_postfix_str(f'Iterations: {max_iters} Success: {success} Closest: {new_node_pos} End: {end_pos}')

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

    def add_example(self, env_data, graph, adj_list, costs, opt_path, dist, prev):
        self.instances.append((env_data, graph, adj_list, costs, opt_path, dist, prev))

    def get(self, i):
        return self.instances[i]

    def len(self):
        return len(self.instances)


class EvaluationDataset:
    def __init__(self, envs):
        self.envs = envs


def make_data(name, env, n_instances):
    dataset = CustomDataset()
    pbar = tqdm(range(n_instances))

    for _ in pbar:
        dataset.add_example(*env.rgg(force_path=True))
        pbar.set_postfix_str(f'Generating RGGs')
        env.reset()

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


def train(train_path, model_name, epochs=100, fast=False, mode='2d'):
    if mode == '2d':
        env = Scatter2D(width=10, height=10)
    elif mode == '6d':
        env = RobotEnv()

    model = Model(in_dim=4 + env.n_degrees).to(device)
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
                env_data, graph, adj_list, costs, opt_path, dist, prev = dataset.get(idx)

                env.load_data(env_data)

                with torch.no_grad():
                    graph = Data(graph.x.to(device), graph.edge_index.to(device))
                    E = model(graph.x, graph.edge_index).squeeze().cpu()  # [n_nodes, n_nodes]

                    _, _, _, success, steps = helper(torch.clone(E), env, graph, explore_steps=1000000)
                    assert success
                    tree_nodes, tree_edges, frontier, _, _ = helper(torch.clone(E), env, graph,
                                                                    explore_steps=random.randint(0, steps - 1), train_mode=True)

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

                    #print(target_edges)

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
                    env_data, graph, adj_list, costs, opt_path, dist, prev = dataset.get(instance)
                    env.load_data(env_data)

                    # start_time = time.process_time()
                    E = model(graph.x.to(device), graph.edge_index.to(device)).squeeze().cpu() # [n_nodes, n_nodes]

                    _, _, _, success, steps = helper(torch.clone(E), env, graph, explore_steps=1000000)
                    assert success
                    tree_nodes, tree_edges, frontier, _, _ = helper(torch.clone(E), env, graph,
                                                                    explore_steps=random.randint(0, steps - 1), train_mode=True)

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


def test_rgg():
    env_2d = Scatter2D(width=10, height=10, p=0.08)
    env_data, graph, adj_list, costs, opt_path, dist, prev = env_2d.rgg(force_path=True)

    model = Model(in_dim=6)
    model.load_state_dict(torch.load('models/fast-epochs=10-buffer=1024-batch=32.pth'))

    E = model(graph.x, graph.edge_index).squeeze()  # [n_nodes, n_nodes]
    tree_nodes, tree_edges, frontier, success, steps = helper(E, env_2d, graph, 1000)
    oracle_node = min(tree_nodes, key=lambda x: dist[x])
    oracle_node_next = prev[oracle_node]
    oracle_edge = None

    if oracle_node_next:
        oracle_edge = torch.LongTensor([oracle_node, oracle_node_next])

    env_2d.visualise(graph, special_edges=tree_edges, frontier_edges=frontier, oracle_edge=oracle_edge, opt_path=opt_path)



def train_2d():
    #test_rgg()
    #exit(0)

    model = Model(in_dim=6)
    model.load_state_dict(torch.load('models/good.pth'))

    model2 = Model(in_dim=6)
    model2.load_state_dict(torch.load('models/fast-epochs=10-buffer=1024-batch=32.pth'))

    model_base = Model(in_dim=6)
    model_base.load_state_dict(torch.load('models/model_random.pth'))

    # world_empty = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # ]
    #
    # world_basic = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    # ]
    #
    # world_checker = [
    #     1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # ]
    #
    # world_x = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    #     0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    #     0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    #     0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
    #     0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    #     0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # ]
    #
    # world_corridor = [
    #     0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #     0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    # ]
    #
    # world_scatter = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #     0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    #     0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    # ]

    #make_data(TRAIN_SET, Scatter2D(width=10, height=10, p=0.08), 1000)
    #train(f'objs/{TRAIN_SET}.pkl', 'good', 10, fast=True)

    env_eval = Scatter2D(width=10, height=10, force_dist=5, p=0.16)
    env_data = []
    for _ in range(1000):
        env_eval.reset()
        env_data.append(env_eval.get_data())
    evaluate(env_eval, env_data, model, rnd=True)
    evaluate(env_eval, env_data, model)
    evaluate_rrt(env_eval, env_data)

def train_6d():
    #make_data('robots-300', RobotEnv(), n_instances=300)
    #train(f'objs/robots-300.pkl', 'robot', 300, fast=True, mode='6d')

    model = Model(in_dim=4 + 6)
    model.load_state_dict(torch.load('models/robot.pth'))

    env_eval = RobotEnv()
    env_data = []
    for _ in range(100):
        env_eval.reset()
        env_data.append(env_eval.get_data())

    #evaluate(env_eval, env_data, model, rnd=True)
    #evaluate(env_eval, env_data, model)
    evaluate_rrt(env_eval, env_data)

if __name__ == '__main__':

    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['figure.figsize'] = (6, 6)

    train_6d()


#%%
