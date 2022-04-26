import os
import sys
from contextlib import contextmanager

import pybullet
import random
from new import N_SAMPLES, K, dijkstra
import torch
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pybullet_planning import CIRCULAR_LIMITS, joints_from_names, get_custom_limits, \
    get_collision_fn, load_pybullet, create_box, set_pose, Pose, Point, Euler, get_movable_joints, connect
import pybullet as p
import time

ROBOT_PATH = 'universal_robot/ur_description/urdf/ur5.urdf' # 'eth_rfl_robot/eth_rfl_description/urdf/eth_rfl.urdf'

@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


class Env:
    def __init__(self, n_degrees, force_dist=0):
        self.n_degrees = n_degrees
        self.k = K
        self.n_samples = N_SAMPLES
        self.upper_bounds = None
        self.lower_bounds = None
        self.start = None
        self.end = None
        self.force_dist = force_dist

        self.reset()

    def get_data(self):
        return (self.start, self.end, self.obstacle_data())

    def obstacle_data(self):
        raise NotImplementedError()

    def load_data(self, env_data):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def set_bounds(self):
        raise NotImplementedError()

    def set_start_end(self):
        while True:
            self.start = self.rand_position()
            self.end = self.rand_position()
            if (self.start - self.end).pow(2).sum().pow(0.5) < self.force_dist:
                continue
            if self.not_in_obstacle(self.start) and self.not_in_obstacle(self.end):
                break

    def rand_position(self):
        return self.lower_bounds + torch.rand(self.n_degrees) * (self.upper_bounds - self.lower_bounds)

    def rgg(self, force_path=False):
        for att in range(10):
            if att > 5:
                print('resetting')
                self.reset()

            graph = self._rgg()
            graph, adj_list, costs, opt_path, dist, prev = self.graph_properties(graph)

            if opt_path == [] and force_path:
                continue
            return (self.start, self.end, self.obstacle_data()), graph, adj_list, costs, opt_path, dist, prev
        raise Exception()

    def _rgg(self):
        rgg_size = self.n_samples + 2

        r = self.lower_bounds + torch.rand((self.n_samples, self.n_degrees)) * (
                    self.upper_bounds - self.lower_bounds)  # [[x1,y1],[x2,y2],...,[xn,yn]]

        r = torch.row_stack([r, self.start, self.end])

        l2 = (r - self.end).pow(2).sum(dim=1)
        r = torch.column_stack([r, l2])

        # check if each point is in free space or obstacle or goal
        one_hot = []
        for idx, cfg in enumerate(r[:, :self.n_degrees]):
            if idx == r.shape[0] - 1:
                one_hot.append([0, 0, 1])
            # elif (cfg[0].item(), cfg[1].item()) in self.obstacles:
            #     one_hot.append([0, 1, 0])
            # else:
            #     one_hot.append([1, 0, 0])
            elif not self.not_in_obstacle(cfg):
                one_hot.append([0, 1, 0])
            else:
                one_hot.append([1, 0, 0])
        r = torch.column_stack([r, torch.tensor(one_hot)])

        D = (r[:, 0][:, None] - r[:, 0].repeat(rgg_size, 1)).pow(2)
        for deg in range(1, self.n_degrees):
            D += (r[:, deg][:, None] - r[:, deg].repeat(rgg_size, 1)).pow(2)

        D += torch.eye(rgg_size) * (2 ** 20)
        knn = D.argsort(dim=1)[:, :self.k]

        us = torch.arange(rgg_size).repeat(1, self.k).squeeze()
        vs = knn.T.reshape(-1, 1).squeeze()

        edge_index = to_undirected(torch.row_stack([us, vs]))
        return Data(x=r, edge_index=edge_index)

    def graph_properties(self, graph):
        adj_list = defaultdict(list)
        for [u, v] in graph.edge_index.T:
            adj_list[u.item()].append(v.item())
        start_node = graph.x.shape[0] - 2
        goal_node = graph.x.shape[0] - 1

        costs = {}
        pos = graph.x[:, :self.n_degrees]
        for node in range(graph.x.shape[0]):
            for nb in adj_list[node]:
                if self.not_collide(pos[nb], pos[node]):
                    dist = (pos[nb] - pos[node]).pow(2).sum().pow(0.5).item()
                else:
                    dist = float('inf')
                costs[(node, nb)] = dist

        opt_path, dist, prev = dijkstra(adj_list, costs, start_node, goal_node)

        return graph, adj_list, costs, opt_path, dist, prev

    def gen(self):
        raise NotImplementedError()

    def not_in_obstacle(self, pos):
        raise NotImplementedError()

    def not_collide(self, pos1, pos2):
        raise NotImplementedError()


class RobotEnv(Env):
    def __init__(self, **kwargs):
        connect(use_gui=False)
        with suppress_stdout():
            self.robot = load_pybullet(ROBOT_PATH, fixed_base=True)

        # arm_joint_names = ['gantry_x_joint',
        #                    '{}_gantry_y_joint'.format('r'),
        #                    '{}_gantry_z_joint'.format('r')] + \
        #                   ['{}_robot_joint_{}'.format('r', i + 1) for i in range(6)]
        # self.arm_joints = joints_from_names(self.robot, arm_joint_names)

        self.arm_joints = get_movable_joints(self.robot)
        self.obstacles = []
        self.obstacle_raw = []

        super().__init__(n_degrees=6, **kwargs)

    def obstacle_data(self):
        return self.obstacle_raw

    def load_data(self, env_data):
        self.start = env_data[0]
        self.end = env_data[1]
        self.obstacles = []
        self.obstacle_raw = env_data[2]
        for [x, y, z, w, h, d] in self.obstacle_raw:
            block1 = create_box(w, h, d)
            set_pose(block1, Pose(Point(x, y, z), Euler(yaw=np.pi / 2)))
            self.obstacles.append(block1)
        self.collision_fn = get_collision_fn(self.robot, self.arm_joints, obstacles=self.obstacles,
                                                 self_collisions=False)

    def reset(self):
        for obs in self.obstacles:
            p.removeBody(obs)
        self.obstacles = []
        self.obstacle_raw = []
        self.set_bounds()
        self.gen()
        self.set_start_end()

    def set_bounds(self):
        lower_limits, upper_limits = get_custom_limits(self.robot, self.arm_joints, circular_limits=CIRCULAR_LIMITS)
        self.lower_bounds = torch.tensor(lower_limits)
        self.upper_bounds = torch.tensor(upper_limits)

    def gen(self):
        for _ in range(5):
            while True:
                x, y, z, w, h, d = random.random() - 0.5, random.random() - 0.5, random.random() - 0.5, \
                                   0.2 + random.random() * 0.1, 0.2 + random.random() * 0.1, 0.2 + random.random() * 0.1
                if abs(z) < .35:
                    continue
                break

            block = create_box(w, h, d)
            set_pose(block, Pose(Point(x, y, z), Euler(yaw=np.pi / 2)))
            self.obstacles.append(block)
            self.obstacle_raw.append([x, y, z, w, h, d])
        self.collision_fn = get_collision_fn(self.robot, self.arm_joints, obstacles=self.obstacles,
                                        self_collisions=False)

    def not_in_obstacle(self, pos):
        return not self.collision_fn(pos.cpu())

    def not_collide(self, pos1, pos2):
        assert len(pos1) == len(pos2) == self.n_degrees

        if not self.not_in_obstacle(pos1) or not self.not_in_obstacle(pos2):
            return False

        vec = pos2 - pos1
        d = np.linalg.norm(vec.cpu())

        n_steps = int(d / 0.25)

        #t1 = time.time()

        for k in range(0, n_steps):
            c = pos1 + k * 1. / n_steps * vec
            if not self.not_in_obstacle(c):
                return False

        #raise Exception(time.time() - t1)

        return True


class Env2D(Env):
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height
        self.obstacles = set()
        super().__init__(n_degrees=2, **kwargs)

    def obstacle_data(self):
        return self.obstacles

    def load_data(self, env_data):
        self.start = env_data[0]
        self.end = env_data[1]
        self.obstacles = env_data[2]

    def reset(self):
        self.obstacles = set()
        self.set_bounds()
        self.gen()
        self.set_start_end()

    def set_bounds(self):
        self.lower_bounds = torch.tensor([0, 0])
        self.upper_bounds = torch.tensor([self.width, self.height])

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

        ax.scatter(px_free, py_free, c='lime', alpha=0.5, label='Free nodes')
        ax.scatter(px_collide, py_collide, c='red', zorder=2, label='Unfree nodes')

        us, vs = graph.edge_index
        lines = [(pos[us[i].item()], pos[vs[i].item()]) for i in range(graph.edge_index.shape[1])]
        lc = matplotlib.collections.LineCollection(lines, colors='green', linewidths=2, alpha=0.2, label='RGG edges')
        ax.add_collection(lc)

        if opt_path:
            path_lines = []
            for u, v in opt_path:
                path_lines.append((pos[u], pos[v]))
            lc2 = matplotlib.collections.LineCollection(path_lines, colors='blue', linewidths=6, label='Plan')
            ax.add_collection(lc2)

        lines = [(pos[u.item()], pos[v.item()]) for [u, v] in special_edges.T]
        lc = matplotlib.collections.LineCollection(lines, colors='magenta', linewidths=3, label='Tree')
        ax.add_collection(lc)

        lines = [(pos[u.item()], pos[v.item()]) for [u, v] in frontier_edges.T]
        lc = matplotlib.collections.LineCollection(lines, colors='cyan', linewidths=3, alpha=0.5, label='Frontier')
        ax.add_collection(lc)

        if oracle_edge is not None:
            lines = [(pos[oracle_edge[0].item()], pos[oracle_edge[1].item()])]
            lc = matplotlib.collections.LineCollection(lines, colors='yellow', linewidths=6)
            ax.add_collection(lc)

        ax.scatter([pos[-2, 0]], [pos[-2, 1]], marker='s', s=120, c='gold', zorder=2, label='Start')
        ax.scatter([pos[-1, 0]], [pos[-1, 1]], marker='*', s=250, c='gold', zorder=2, label='End')

        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])

        ax.autoscale()
        ax.margins(0.1)
        #plt.subplots_adjust(right=0.7)
        #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig("11.png", bbox_inches="tight")
        plt.show()


class Scatter2D(Env2D):
    def __init__(self, map=None, p=0.1, *args, **kwargs):
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

