"""
RRT_star 2D
@author: huiming zhou
"""

import math
import numpy as np
import collections
from typing import Sequence
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftState
from shapely.geometry import LineString, Polygon
from pdm4ar.exercises.final21.plot_path import Plotting


class QueueFIFO:
    """
    Class: QueueFIFO
    Description: QueueFIFO is designed for First-in-First-out rule.
    """

    def __init__(self):
        self.queue = collections.deque()

    def empty(self):
        return len(self.queue) == 0

    def put(self, node):
        self.queue.append(node)  # enter from back

    def get(self):
        return self.queue.popleft()  # leave from front


class Node:
    def __init__(self, n: Sequence):
        assert len(n) == 2
        self.x = n[0]
        self.y = n[1]
        self.psi = None
        self.parent = None
        self.child = None

    @classmethod
    def from_state(cls, n: SpacecraftState):
        return Node([n.x, n.y])

    def point_to(self, n: "Node"):
        dx = n.x - self.x
        dy = n.y - self.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def from_point(self, n: "Node"):
        dx = self.x - n.x
        dy = self.y - n.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def equal(self, b: "Node"):
        return True if self.x == b.x and self.y == b.y else False

    def is_bound_in_range(self, x_range: Sequence, y_range: Sequence):
        assert len(x_range) == 2 and len(y_range) == 2
        x_range = sorted(x_range)
        y_range = sorted(y_range)
        return True if x_range[0] < self.x < x_range[1] and y_range[0] < self.y < y_range[1] else False


class RrtStar:
    def __init__(self, x_start, x_goal, static_obstacles: Sequence[StaticObstacle], visualize=False,
                 step_len=10, goal_sample_rate=0.1, search_radius=20, iter_max=2000, safe_offset=3.0, safe_boarder=3.0):
        self.s_start = x_start
        self.s_goal = x_goal
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.plotting = Plotting(x_start, x_goal)
        self.visualize = visualize
        self.path = []
        self.static_obstacles = static_obstacles
        self.offset = safe_offset
        self.safe_boarder = safe_boarder
        # environment boarder
        env = self.static_obstacles[0].shape
        self.x_range = [env.bounds[0], env.bounds[2]]
        self.y_range = [env.bounds[1], env.bounds[3]]
        self.safe_s_obstacles = {0.0: self.static_obstacles,
                                 2.0: self.get_safe_obstacles(2.0),
                                 3.0: self.get_safe_obstacles(3.0)}

    def get_safe_env_bound(self, offset=3.0):
        x_range = [self.x_range[0] + offset, self.x_range[1] - offset]
        y_range = [self.y_range[0] + offset, self.y_range[1] - offset]
        return x_range, y_range

    def get_safe_obstacles(self, offset=2.0):
        safe_s_obstacles = []
        # minx, maxx = self.x_range[0] + offset, self.x_range[1] - offset
        # miny, maxy = self.y_range[0] + offset, self.y_range[1] - offset
        # safe_boundary = LineString([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
        # safe_s_obstacles.append(safe_boundary)
        for s_obstacle in self.static_obstacles[1:]:
            safe_boundary = s_obstacle.shape.buffer(offset, resolution=16, join_style=2, mitre_limit=1).exterior
            safe_s_obstacles.append(safe_boundary)
        return safe_s_obstacles

    def is_collision(self, node_near, node_new, offset=None):
        if offset is not None:
            if offset not in self.safe_s_obstacles.keys():
                self.safe_s_obstacles.update({offset: self.get_safe_obstacles(offset)})
        else:
            if self.offset not in self.safe_s_obstacles.keys():
                self.safe_s_obstacles.update({self.offset: self.get_safe_obstacles(self.offset)})
            offset = self.offset
        if offset:
            for s_obstacle in self.safe_s_obstacles[offset]:
                path = LineString([(node_near.x, node_near.y), (node_new.x, node_new.y)])
                if path.intersects(s_obstacle):
                    return True
        else:
            for s_obstacle in self.safe_s_obstacles[offset]:
                path = LineString([(node_near.x, node_near.y), (node_new.x, node_new.y)])
                if path.intersects(s_obstacle.shape.convex_hull.exterior):
                    return True
        return False

    def planning(self):
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if k % 500 == 0:
                print(f"[planning] {k} / {self.iter_max}")

            if node_new and not self.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()
        self.path = self.extract_path(self.vertex[index])

        if self.visualize:
            self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]
        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                         if not self.is_collision(self.vertex[i], self.s_goal)]
            if len(cost_list) > 0:
                return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.safe_boarder

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.is_collision(node_new, self.vertex[ind])]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def update_cost(self, parent_node):
        OPEN = QueueFIFO()
        OPEN.put(parent_node)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.child) == 0:
                continue

            for node_c in node.child:
                node_c.Cost = self.get_new_cost(node, node_c)
                OPEN.put(node_c)

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)