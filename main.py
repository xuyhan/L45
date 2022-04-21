# adapted from https://gist.github.com/Fnjn/58e5eaa27a3dc004c3526ea82a92de80


import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import deque
import torch
from torch_geometric.data import Data, Dataset, DataLoader

class Line():
    """ Define line """
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn


def Intersection(line, center, radius):
    a = np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a);
    t2 = (-b - np.sqrt(discriminant)) / (2 * a);

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True



def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def isInObstacle(vex, obstacles, radius):
    for obs in obstacles:
        if distance(obs, vex) < radius:
            return True
    return False


def isThruObstacle(line, obstacles, radius):
    for obs in obstacles:
        if Intersection(line, obs, radius):
            return True
    return False


def nearest(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min (stepSize, length)

    newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
    return newvex


def window(startpos, endpos):
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height


def isInWindow(pos, winx, winy, width, height):
    if winx < pos[0] < winx+width and \
        winy < pos[1] < winy+height:
        return True
    else:
        return False


class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

        self.x = [[startpos[0], startpos[1], 0, 1, 0],
                  [endpos[0], endpos[1], 0, 0, 1]]
        self.edge_index = [[], []]

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []

            self.x.append([pos[0], pos[1], 1, 0, 0])

        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

        self.edge_index[0].append(idx1)
        self.edge_index[1].append(idx2)

    def randomPosition(self):
        rx = random()
        ry = random()

        posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        return posx, posy

    def to_data(self):
        return torch.tensor(self.x).float(), torch.LongTensor(self.edge_index)


def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            #print('success')
            # break
    return G


def RRT_GNN(startpos, endpos, obstacles, n_iter, radius, stepSize, model):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        for t in range(10):
            #x, edge_index = G.to_data()
            #pred = model(x, edge_index, batch=torch.LongTensor([0] * x.shape[0])).squeeze()
            #randvex = (pred[0].item(), pred[1].item())

            randvex = G.randomPosition()
            if t == 9:
                raise Exception()

            if isInObstacle(randvex, obstacles, radius):
                continue
            break

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        G.distances[newidx] = G.distances[nearidx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            line = Line(vex, newvex)
            if isThruObstacle(line, obstacles, radius):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

            G.success = True

        yield G

def RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize, data_store):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        ###
        x, edge_index = G.to_data()
        ###

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        G.distances[newidx] = G.distances[nearidx] + dist

        y = torch.tensor([newvex[0], newvex[1]]).float()

        data = Data(x, edge_index, y=y)
        data_store.append(data)

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            line = Line(vex, newvex)
            if isThruObstacle(line, obstacles, radius):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

            G.success = True

    return G

def dijkstra(G):
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)



def plot(fig, ax, G, obstacles, radius, path=None):
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]

    for obs in obstacles:
        circle = plt.Circle(obs, radius, color='red')
        ax.add_artist(circle)

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)

    plt.draw()
    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path

import torch_geometric
import networkx as nx
from torch_geometric.nn import GCNConv, Linear, GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


def visualise_graph(graph):
    G = torch_geometric.utils.to_networkx(graph)
    nx.draw(G, alpha=.8, arrows=True)

class PathFinder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(5, 32))
        n_layers = 4

        for i in range(n_layers - 1):
            self.convs.append(GATConv(32, 32))

        self.linear = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    startpos = (0., 0.) #(random() * 10, random() * 10)  # (0., 0.)
    endpos = (4, 4.) #(random() * 10, random() * 10)  # (10., 10.)
    obstacles = []  # [(1., 1.), (2., 2.)]
    n_iter = 500
    radius = 0.5
    stepSize = 0.7

    model = PathFinder()
    model.load_state_dict(torch.load('models/model.pth', map_location=torch.device('cpu')))

    fig, ax = plt.subplots()

    for G in RRT_GNN(startpos, endpos, obstacles, n_iter, radius, stepSize, model):
        plot(fig, ax, G, obstacles, radius)

    # test_g = data_store[40]
    # visualise_graph(test_g)

    if G.success:
        print('Success')
        path = dijkstra(G)
        print(path)
        plot(fig, ax, G, obstacles, radius, path)
    else:
        print('Failure')
        plot(fig, ax, G, obstacles, radius)

if __name__ == '_ _main__':
    from tqdm import tqdm

    data_store = []
    model = PathFinder()
    N_RUNS = 200
    N_EPOCHS = 100

    for _ in tqdm(range(N_RUNS)):
        startpos = (random() * 10, random() * 10) #(0., 0.)
        endpos = (random() * 10, random() * 10) #(10., 10.)
        obstacles = [] #[(1., 1.), (2., 2.)]
        n_iter = 200
        radius = 0.5
        stepSize = 0.7

        G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize, data_store)

        # test_g = data_store[40]
        # visualise_graph(test_g)

        # if G.success:
        #     path = dijkstra(G)
        #     print(path)
        #     plot(G, obstacles, radius, path)
        # else:
        #     plot(G, obstacles, radius)

    train_data = data_store[:-1000]
    test_data = data_store[-1000:]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    min_train_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            out = model.forward(batch.x, batch.edge_index, batch.batch)
            batch_size = out.shape[0]
            loss = torch.nn.MSELoss()(out, batch.y.reshape(batch_size, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            del batch

        train_loss /= len(train_data)
        print('Epoch %s Train Loss %s' % (epoch, train_loss))

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), 'models/model.pth')

        test_loss = 0

        for batch in test_loader:
            batch = batch.to(device)
            with torch.no_grad():
                out = model.forward(batch.x, batch.edge_index, batch.batch)
                batch_size = out.shape[0]
                loss = torch.nn.MSELoss()(out, batch.y.reshape(batch_size, 2))
                test_loss += loss.item()

        test_loss /= len(test_data)

        print('Epoch %s Test Loss %s' % (epoch, test_loss))
#%%
