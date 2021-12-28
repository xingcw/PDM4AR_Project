import os
import sys
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString

from plot_path import Plotting
import dubins_path as dubins
import draw as draw

class Node:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent = None
        self.cost = 0.0
        self.path_x = []
        self.path_y = []
        self.path_yaw = []

class DubinsRRTStar:
    def __init__(self, sx, sy, syaw, gx, gy, gyaw, vehicle_radius, step_len,
                 goal_sample_rate, search_radius, iter_max, dg_scenario):
        self.s_start = Node(sx, sy, syaw)
        self.s_goal = Node(gx, gy, gyaw)
        self.vr = vehicle_radius
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.curv = 1
        self.plotting = Plotting([sx,sy], [gx,gy])
        self.env = dg_scenario
        # self.utils = utils.Utils()

        self.fig, self.ax = plt.subplots()
        self.delta = 0.5    # you can modify the value
        self.x_range = [0, 100]
        self.y_range = [0, 100]
        # self.obs_circle = self.obs_circle()
        # self.obs_boundary = self.env.obs_boundary
        # self.utils.update_obs(self.obs_circle, self.obs_boundary, [])

        self.V = [self.s_start]
        self.path = None

    def is_collision(self, node):
        for s_obstacle in list(self.env.static_obstacles.values())[1:]:
            for i in range(len(node.path_x)-1):
                path = LineString([(node.path_x[i], node.path_y[i]), (node.path_x[i+1], node.path_y[i+1])])
                if path.intersects(s_obstacle.shape.convex_hull):
                    return True
        return False

    def planning(self):
        for i in range(self.iter_max):
            print("Iter:", i, ", number of nodes:", len(self.V))
            rnd = self.Sample()     # sample
            node_nearest = self.Nearest(self.V, rnd)
            new_node = self.Steer(node_nearest, rnd)

            if new_node and not self.is_collision(new_node):
                near_indexes = self.Near(self.V, new_node)
                new_node = self.choose_parent(new_node, near_indexes)

                if new_node:
                    self.V.append(new_node)
                    self.rewire(new_node, near_indexes)

        last_index = self.search_best_goal_node()
        print(last_index)
        path = self.generate_final_course(last_index)
        print("get!")
        px = [s[0] for s in path]
        py = [s[1] for s in path]

    def Sample(self):
        delta = 0.5 # should be the diameter of the vehicle

        if random.random() > self.goal_sample_rate:
            return Node(random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                        random.uniform(-math.pi, math.pi))
        else:
            return self.s_goal

    @staticmethod
    def Nearest(nodelist, n):
        # return the nearest node
        return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
                                       for nd in nodelist]))]

    def Steer(self, node_start, node_end):
        # create dubins path from node_start to node_end
        sx, sy, syaw = node_start.x, node_start.y, node_start.yaw
        gx, gy, gyaw = node_end.x, node_end.y, node_end.yaw
        maxc = self.curv

        path = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc)

        if len(path.x) <= 1:
            return None

        node_new = Node(path.x[-1], path.y[-1], path.yaw[-1])
        node_new.path_x = path.x
        node_new.path_y = path.y
        node_new.path_yaw = path.yaw
        node_new.cost = node_start.cost + path.L
        node_new.parent = node_start

        return node_new

    def Near(self, nodelist, node):
        n = len(nodelist) + 1
        r = min(self.search_radius * math.sqrt((math.log(n)) / n), self.step_len)

        dist_table = [(nd.x - node.x) ** 2 + (nd.y - node.y) ** 2 for nd in nodelist]
        node_near_ind = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r ** 2]

        return node_near_ind

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.V[i]
            t_node = self.Steer(near_node, new_node)
            if t_node and not self.is_collision(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.Steer(self.V[min_ind], new_node)

        return new_node

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.get_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.V[i]
            edge_node = self.Steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = ~self.is_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                self.V[i] = edge_node
                self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.V:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.V]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.step_len]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.Steer(self.V[goal_ind], self.s_goal)
            if t_node and not self.is_collision(t_node):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.V[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.V[i].cost == min_cost:
                return i

        return None

    def calc_dist_to_goal(self, x, y):
        dx = x - self.s_goal.x
        dy = y - self.s_goal.y
        return math.hypot(dx, dy)

    def generate_final_course(self, goal_index):
        # return the final path
        print("final")
        path = [[self.s_goal.x, self.s_goal.y]]
        node = self.V[goal_index]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            node = node.parent
        path.append([self.s_start.x, self.s_start.y])
        return path