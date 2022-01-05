import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random
import math
from typing import Sequence
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftState
from shapely.geometry import LineString, Polygon


def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def ortho(vect2d):
    """Computes an orthogonal vector to the one given"""
    return np.array((-vect2d[1], vect2d[0]))


def dist(pt_a, pt_b):
    """Euclidian distance between two (x, y) points"""
    return ((pt_a[0]-pt_b[0])**2 + (pt_a[1]-pt_b[1])**2)**.5


class RRT:
    def __init__(self,  static_obstacles: Sequence[StaticObstacle], precision=(5, 5, 1)):
        self.nodes = {}
        self.edges = {}
        self.local_planner = Dubins(4, 1)
        self.x_range = [0, 100]
        self.y_range = [0, 100]
        self.goal = (0, 0, 0)
        self.root = (0, 0, 0)
        self.precision = precision
        # environment boarder
        self.static_obstacles = static_obstacles
        env = self.static_obstacles[0].shape

        self.x_range = [env.bounds[0], env.bounds[2]]
        self.y_range = [env.bounds[1], env.bounds[3]]

    def set_start(self, start):
        self.nodes = {}
        self.edges = {}
        self.nodes[start] = Node(start, 0, 0)
        self.root = start

    def sample(self):
        delta = 0.5  # should be the diameter of the vehicle

        x = random.uniform(self.x_range[0] + delta, self.x_range[1] - delta)
        y = random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)
        return x, y, np.random.rand()*np.pi*2

    def is_free(self, point_start, point_end):
        for s_obstacle in list(self.static_obstacles)[1:]:
            path = LineString([(point_start[0], point_start[1]), (point_end[0], point_end[1])])
            if path.intersects(s_obstacle.shape.convex_hull):
                return False
            if point_start[0] < self.x_range[0] or point_start[0] > self.x_range[1]:
                return False
            if point_start[1] < self.y_range[0] or point_start[1] > self.y_range[1]:
                return False
        return True

    def run(self, goal, nb_iteration=100, goal_rate=.1, metric='local'):
        assert len(goal) == len(self.precision)
        self.goal = goal

        for _ in range(nb_iteration):
            if np.random.rand() > 1 - goal_rate:
                sample = goal
            else:
                sample = self.sample()

            options = self.select_options(sample, 10, metric)

            for node, option in options:
                if option[0] == float('inf'):
                    break
                path = self.local_planner.generate_points(node, sample, option[1], option[2])

                for i in range(len(path)-1):
                    if not self.is_free(path[i], path[i+1]):
                        break
                else:
                    # Adding the node
                    # To compute the time, we use a constant speed of 1 m/s
                    # As the cost, we use the distance
                    self.nodes[sample] = Node(sample,self.nodes[node].time+option[0], self.nodes[node].cost+option[0])
                    self.nodes[node].destination_list.append(sample)
                    # Adding the Edge
                    self.edges[node, sample] = Edge(node, sample, path, option[0])
                    if self.in_goal_region(sample):
                        return
                    break

    def select_options(self, sample, nb_options, metric='local'):
        if metric == 'local':
            options = []
            for node in self.nodes:
                options.extend(
                    [(node, opt)\
                     for opt in self.local_planner.all_options(node, sample)])
            # sorted by cost
            options.sort(key=lambda x: x[1][0])
            options = options[:nb_options]
        else:
            options = [(node, dist(node, sample)) for node in self.nodes]
            options.sort(key=lambda x: x[1])
            options = options[:nb_options]
            new_opt = []
            for node, _ in options:
                db_options = self.local_planner.all_options(node, sample)
                new_opt.append((node, min(db_options, key=lambda x: x[0])))
            options = new_opt
        return options

    def in_goal_region(self, sample):
        for i, value in enumerate(sample):
            if abs(self.goal[i]-value) > self.precision[i]:
                return False
        return True

    def select_best_edge(self):
        node = max([(child, self.children_count(child))\
                    for child in self.nodes[self.root].destination_list],
                   key=lambda x: x[1])[0]
        best_edge = self.edges[(self.root, node)]
        for child in self.nodes[self.root].destination_list:
            if child == node:
                continue
            self.edges.pop((self.root, child))
            self.delete_all_children(child)
        self.nodes.pop(self.root)
        self.root = node
        return best_edge

    def delete_all_children(self, node):
        if self.nodes[node].destination_list:
            for child in self.nodes[node].destination_list:
                self.edges.pop((node, child))
                self.delete_all_children(child)
        self.nodes.pop(node)

    def children_count(self, node):
        if not self.nodes[node].destination_list:
            return 0
        total = 0
        for child in self.nodes[node].destination_list:
            total += 1 + self.children_count(child)
        return total

    def get_final_path(self):
        a = self.goal
        b = self.goal
        path_x = []
        path_y = []
        psi = []
        while a != self.root:
            for _, val in self.edges.items():
                if val.node_to == b:
                    path = np.array(val.path)
                    b = val.node_from
                    path_x.append(np.flip(path[1:, 0]))
                    path_y.append(np.flip(path[1:, 1]))
                    if b == self.root:
                        a = self.root
        path_x = np.concatenate(path_x)
        path_x = np.append(path_x, self.root[0])
        path_y = np.concatenate(path_y)
        path_y = np.append(path_y, self.root[1])
        path_x = np.flip(path_x)
        path_y = np.flip(path_y)
        for i in range(len(path_x) - 1):
            psi.append(math.atan2(path_y[i + 1] - path_y[i], path_x[i + 1] - path_x[i]))
        psi = np.asarray(psi)
        path_x = path_x[:-1]
        path_y = path_y[:-1]
        # path = np.vstack((path_x.T,path_y.T))
        # path = path.T
        # velo = path[1:] - path[:-1]
        # velo = velo / np.tile(np.linalg.norm(velo, axis=1, ord=2), (2, 1)).T
        # accl = velo[1:] - velo[:-1]
        # accl = accl / np.tile(np.linalg.norm(accl, axis=1, ord=2), (2, 1)).T
        return path_x, path_y, psi

    def plot(self, file_name='', close=False, nodes=False):
        if nodes and self.nodes:
            nodes = np.array(list(self.nodes.keys()))
            plt.scatter(nodes[:, 0], nodes[:, 1])
            plt.scatter(self.root[0], self.root[1], c='g')
            plt.scatter(self.goal[0], self.goal[1], c='r')
        path_x, path_y, psi = self.get_final_path()
        plt.plot(path_x, path_y,'r')
        if file_name:
            plt.savefig(file_name)
        if close:
            plt.close()


class Dubins:
    def __init__(self, radius, point_separation):
        assert radius > 0 and point_separation > 0
        self.radius = radius
        self.point_separation = point_separation

    def all_options(self, start, end, sort=False):
        center_0_left = self.find_center(start, 'L')
        center_0_right = self.find_center(start, 'R')
        center_2_left = self.find_center(end, 'L')
        center_2_right = self.find_center(end, 'R')
        options = [self.lsl(start, end, center_0_left, center_2_left),
                   self.rsr(start, end, center_0_right, center_2_right),
                   self.rsl(start, end, center_0_right, center_2_left),
                   self.lsr(start, end, center_0_left, center_2_right),
                   self.rlr(start, end, center_0_right, center_2_right),
                   self.lrl(start, end, center_0_left, center_2_left)]
        if sort:
            options.sort(key=lambda x: x[0])
        return options

    def find_center(self, point, side):
        """
        point : tuple
        In the form (x, y, psi), with psi in radians.
        The representation of the initial point.
        """
        assert side in 'LR'
        angle = point[2] + (np.pi/2 if side == 'L' else -np.pi/2)
        return np.array((point[0] + np.cos(angle)*self.radius,
                         point[1] + np.sin(angle)*self.radius))

    def lsl(self, start, end, center_0, center_2):
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (end[2]-alpha)%(2*np.pi)
        beta_0 = (alpha-start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, beta_2, straight_dist), True)

    def rsr(self, start, end, center_0, center_2):
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (-end[2]+alpha)%(2*np.pi)
        beta_0 = (-alpha+start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, -beta_2, straight_dist), True)

    def rsl(self, start, end, center_0, center_2):
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = -(psia+alpha-start[2]-np.pi/2)%(2*np.pi)
        beta_2 = (np.pi+end[2]-np.pi/2-alpha-psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, beta_2, straight_dist), True)

    def lsr(self, start, end, center_0, center_2):
        median_point = (center_2 - center_0) / 2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius / half_intercenter)
        beta_0 = (psia - alpha - start[2] + np.pi / 2) % (2 * np.pi)
        beta_2 = (.5 * np.pi - end[2] - alpha + psia) % (2 * np.pi)
        straight_dist = 2 * (half_intercenter ** 2 - self.radius ** 2) ** .5
        total_len = self.radius * (beta_2 + beta_0) + straight_dist
        return (total_len, (beta_0, -beta_2, straight_dist), True)

    def lrl(self, start, end, center_0, center_2):
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0) / 2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if 2 * self.radius < dist_intercenter > 4 * self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2 * np.arcsin(dist_intercenter / (4 * self.radius))
        beta_0 = (psia - start[2] + np.pi / 2 + (np.pi - gamma) / 2) % (2 * np.pi)
        beta_1 = (-psia + np.pi / 2 + end[2] + (np.pi - gamma) / 2) % (2 * np.pi)
        total_len = (2 * np.pi - gamma + abs(beta_0) + abs(beta_1)) * self.radius
        return (total_len,(beta_0, beta_1, 2 * np.pi - gamma),False)

    def rlr(self, start, end, center_0, center_2):
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if 2*self.radius < dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = -((-psia+(start[2]+np.pi/2)+(np.pi-gamma)/2)%(2*np.pi))
        beta_1 = -((psia+np.pi/2-end[2]+(np.pi-gamma)/2)%(2*np.pi))
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len,(beta_0, beta_1, 2*np.pi-gamma), False)

    def generate_points(self, start, end, dubins_path, straight):
        if straight:
            return self.generate_points_straight(start, end, dubins_path)
        return self.generate_points_curve(start, end, dubins_path)

    def generate_points_straight(self, start, end, path):
        total = self.radius*(abs(path[1])+abs(path[0]))+path[2] # Path length
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')

        # We first need to find the points where the straight segment starts
        if abs(path[0]) > 0:
            angle = start[2]+(abs(path[0])-np.pi/2)*np.sign(path[0])
            ini = center_0+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: ini = np.array(start[:2])
        # We then identify its end
        if abs(path[1]) > 0:
            angle = end[2]+(-abs(path[1])-np.pi/2)*np.sign(path[1])
            fin = center_2+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: fin = np.array(end[:2])
        dist_straight = dist(ini, fin)

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius: # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius: # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else: # Straight segment
                coeff = (x-abs(path[0])*self.radius)/dist_straight
                points.append(coeff*fin + (1-coeff)*ini)
        points.append(end[:2])
        return np.array(points)

    def generate_points_curve(self, start, end, path):
        total = self.radius*(abs(path[1])+abs(path[0])+abs(path[2]))
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')
        intercenter = dist(center_0, center_2)
        center_1 = (center_0 + center_2)/2 +\
                   np.sign(path[0])*ortho((center_2-center_0)/intercenter)\
                    *(4*self.radius**2-(intercenter/2)**2)**.5
        psi_0 = np.arctan2((center_1 - center_0)[1],
                           (center_1 - center_0)[0])-np.pi

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius: # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius: # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else: # Middle Turn
                angle = psi_0-np.sign(path[0])*(x/self.radius-abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1+self.radius*vect)
        points.append(end[:2])
        return np.array(points)

    def circle_arc(self, reference, beta, center, x):
        angle = reference[2]+((x/self.radius)-np.pi/2)*np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center+self.radius*vect


class Node:
    def __init__(self, position, time, cost):
        self.destination_list = []
        self.position = position
        self.time = time
        self.cost = cost


class Edge:
    def __init__(self, node_from, node_to, path, cost):
        self.node_from = node_from
        self.node_to = node_to
        self.path = deque(path)
        self.cost = cost

    def finalpath(self):
        return self.path


