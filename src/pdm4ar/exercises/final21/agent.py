from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapely.geometry import LineString, Polygon
from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from pdm4ar.exercises.final21.RRT_star import RrtStar, Node

A_MAX = 10
V_MAX = 50
PSI_DOT_MAX = 2 * np.pi


class Waypoints:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.psi = None
        self.parent = None
        self.child = None

    def get_psi(self):
        if self.child is not None:
            self.psi = np.arctan2(self.child.y - self.y, self.child.x - self.x)

    def point_to(self, waypoint):
        path = [waypoint.x - self.x, waypoint.y - self.y]
        return path


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do NOT modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 sp: SpacecraftGeometry):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.planner = None
        self.waypoints = []
        self.current_goal = None
        self.last_goal = None
        self.current_state = None
        self.visualize = False
        self.name = None
        self.GO = False
        self.ROT = False
        self.STOP = False
        self.max_vx = 0
        self.seed = 4
        self.drift_angle = 0

    def on_episode_init(self, my_name: PlayerName):
        self.name = my_name

    def get_commands(self, sim_obs: SimObservations) -> SpacecraftCommands:
        """ This method is called by the simulator at each time step.

        This is how you can get your current state from the observations:
        my_current_state: SpacecraftState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """

        # todo implement here
        command = SpacecraftCommands(acc_left=0, acc_right=0)
        self.current_state = sim_obs.players[self.name].state
        self.drift_angle = np.arctan2(self.current_state.vy, self.current_state.vx)
        if (self.current_goal is None) or self.STOP:
            command = self.stop()
        else:
            if not self.ready_togo() and self.ROT:
                self.GO = False
                self.ROT = True
                a = self.current_state.psi - self.current_goal.psi + self.current_state.dpsi
                a_l = self.bind_to_range(3 * a, -A_MAX, A_MAX)
                a_r = self.bind_to_range(-3 * a, -A_MAX, A_MAX)
                command = SpacecraftCommands(acc_left=a_l, acc_right=a_r)
                print(f"[reach goal and rotate] target psi: {self.current_goal.psi}")
            elif self.ready_togo() and not self.GO:
                self.GO = True
                self.ROT = False
                # self.update_current_goal(self.get_furthest_no_collision_waypoint())
                print(f"[ready to go] next goal: 'x: {self.current_goal.x}, y: {self.current_goal.y}'")
            elif not self.reach_goal() and self.GO:
                if self.check_future_collision():
                    steer = self.get_steer_direct()
                    if steer == "left":
                        command = SpacecraftCommands(acc_left=-A_MAX, acc_right=-A_MAX/2)
                        print(f"[collision warning] steer to the left!")
                    else:
                        command = SpacecraftCommands(acc_left=-A_MAX/2, acc_right=-A_MAX)
                        print(f"[collision warning] steer to the right!")
                else:
                    delta_s = self.current_goal.point_to(self.current_state)
                    dsa = np.linalg.norm(delta_s)
                    dpsia = 2 * (self.current_state.psi - self.current_goal.psi) + self.current_state.dpsi
                    al = self.bind_to_range(dsa - 2 * self.current_state.vx + dpsia, -A_MAX, A_MAX)
                    ar = self.bind_to_range(dsa - 2 * self.current_state.vx - dpsia, -A_MAX, A_MAX)
                    command = SpacecraftCommands(acc_left=al, acc_right=ar)
                    print(f"[head to goal] goal: 'x: {self.current_goal.x}, y: {self.current_goal.y}' ")
            elif self.reach_goal() and self.GO:
                command = self.stop()

            if self.visualize:
                self.plot_state()
            print(self.current_state)
            print(f"last goal: 'x: {self.last_goal.x}, y: {self.last_goal.y}")
            print(f"current goal: 'x: {self.current_goal.x}, y: {self.current_goal.y}")
            print(f"{command} \n")

        return command

    def update_current_goal(self, goal_waypoint):
        current_point = self.current_state.as_ndarray()[:2]
        if self.current_goal is None:
            self.last_goal = Waypoints(current_point)
        else:
            self.last_goal = self.current_goal
        goal_point = np.array([goal_waypoint.x, goal_waypoint.y])
        path = goal_point - current_point
        self.max_vx = np.sqrt(A_MAX * np.linalg.norm(path))
        self.current_goal = Waypoints(goal_point)
        self.current_goal.psi = np.arctan2(path[1], path[0])

    def reach_goal(self, thresh=5.0):
        if np.abs(self.current_state.x - self.current_goal.x) < thresh and \
                np.abs(self.current_state.y - self.current_goal.y) < thresh:
            return True
        else:
            return False

    def ready_togo(self, thresh=0.05):
        if np.abs(self.current_state.psi - self.current_goal.psi) < thresh and np.abs(self.current_state.dpsi) < thresh:
            return True
        else:
            return False

    def stopped(self, thresh=0.1):
        return np.abs(self.current_state.vx) < thresh and np.abs(self.current_state.vy) < thresh

    def stop(self):
        self.STOP = True
        if np.abs(self.current_state.vx) >= 0.05:
            a_l = self.current_state.vy - self.current_state.vx + 3 * self.current_state.dpsi
            a_r = - self.current_state.vy - self.current_state.vx - 3 * self.current_state.dpsi
            print(f"[stopping] current state: \n {self.current_state}")
            command = SpacecraftCommands(acc_left=a_l, acc_right=a_r)
            return command
        else:
            if self.planner is None:
                np.random.seed(self.seed)
                self.replan()
            self.update_current_goal(self.get_furthest_no_collision_waypoint())
            # if np.linalg.norm(self.last_goal.point_to(self.current_goal)) < 5:
            #     self.replan()
            #     self.update_current_goal(self.get_furthest_no_collision_waypoint())
            self.STOP = False
            self.ROT = True
            self.GO = False
            return SpacecraftCommands(acc_left=0, acc_right=0)

    def get_closest_waypoint(self):
        state = np.array([self.current_state.x, self.current_state.y])
        waypoints = np.array([[waypoint.x, waypoint.y] for waypoint in self.waypoints])
        idx = np.argmin(waypoints - state)
        return self.waypoints[idx]

    def get_furthest_no_collision_waypoint(self):
        if not self.planner.is_collision(self.current_state, self.planner.s_goal):
            return self.waypoints[-1]
        collision = np.array([self.planner.is_collision(self.current_state, waypoint) for waypoint in self.waypoints])
        state = np.array([self.current_state.x, self.current_state.y])
        waypoints = np.array([[waypoint.x, waypoint.y] for waypoint in self.waypoints])
        distance = np.linalg.norm(state - waypoints, axis=1)
        no_collision_dist = np.multiply(distance, ~collision)
        idx = np.argmax(no_collision_dist)
        next_goal = self.waypoints[idx]
        self.waypoints[idx].parent = None
        self.waypoints = self.waypoints[idx:]
        return next_goal

    def get_safe_distance(self):
        v = np.linalg.norm([self.current_state.vx, self.current_state.vy])
        return v ** 2 / (2 * A_MAX)

    def check_future_collision(self, drift_angle=None):
        angle = self.drift_angle + self.current_state.psi if drift_angle is None else drift_angle
        current_point = Node([self.current_state.x, self.current_state.y])
        safe_dist = self.get_safe_distance()
        stop_point = Node([self.current_state.x + safe_dist * np.cos(angle),
                           self.current_state.y + safe_dist * np.sin(angle)])
        return self.planner.is_collision(current_point, stop_point)

    def get_steer_direct(self, angle_dt=0.2):
        psi_l = self.current_state.psi + self.drift_angle + angle_dt
        psi_r = self.current_state.psi + self.drift_angle - angle_dt
        left_collision = self.check_future_collision(psi_l)
        right_collision = self.check_future_collision(psi_r)

        if left_collision:
            return "left"
        elif right_collision:
            return "right"
        elif self.drift_angle + self.current_state.psi > self.current_goal.psi:
            return "right"
        else:
            return "left"

    def replan(self):
        x_start = (self.current_state.x, self.current_state.y)
        x_goal = (self.goal.goal.centroid.x, self.goal.goal.centroid.y)
        self.planner = RrtStar(x_start, x_goal, 10, 0.10, 20, 2000, self.static_obstacles)
        self.planner.planning()
        self.waypoints = []

        for node in reversed(self.planner.path):
            waypoint = Waypoints(node)
            self.waypoints.append(waypoint)
        self.waypoints[0].parent = self.waypoints[0]
        self.waypoints[-1].child = self.waypoints[-1]
        for i, waypoint in enumerate(self.waypoints):
            if waypoint.parent is None:
                waypoint.parent = self.waypoints[i - 1]
            if waypoint.child is None:
                waypoint.child = self.waypoints[i + 1]
        for waypoint in self.waypoints:
            waypoint.get_psi()

    def plot_state(self):
        axs = plt.gca()
        shapely_viz = ShapelyViz(axs)

        # plot original obstacles
        for s_obstacle in self.static_obstacles:
            shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
        shapely_viz.add_shape(self.goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
        axs = shapely_viz.ax
        axs.autoscale()
        axs.set_aspect("equal")

        # plot safe boundary of obstacles
        for safe_s_obstacle in self.planner.safe_s_obstacles:
            axs.plot(*safe_s_obstacle.xy)

        # plot planned path
        axs.plot([x[0] for x in self.planner.path], [x[1] for x in self.planner.path], '-k', linewidth=2)

        # plot velocities
        x_tail = self.current_state.x
        y_tail = self.current_state.y
        dx = np.abs(self.current_state.vx) * np.cos(self.current_state.psi)
        dy = np.abs(self.current_state.vx) * np.sin(self.current_state.psi)
        axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.3, length_includes_head=True, color="r")
        # axs.text(x_tail + 1, y_tail - .4, "$V_x$")

        v = np.linalg.norm(np.array([self.current_state.vx, self.current_state.vy]))
        dx = v * np.cos(self.current_state.psi + self.drift_angle)
        dy = v * np.sin(self.current_state.psi + self.drift_angle)
        axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.3, length_includes_head=True, color="k")
        # axs.text(x_tail + 1, y_tail - .4, "V")

        # dx = np.abs(self.current_state.vy) * np.cos(self.current_state.psi + np.pi / 2)
        # dy = np.abs(self.current_state.vy) * np.sin(self.current_state.psi + np.pi / 2)
        # axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.5, length_includes_head=True, color="k")
        # axs.text(x_tail + 1, y_tail - .4, "$V_y$")

        # plot scan lanes
        angle_dt = 0.2
        dist = self.get_safe_distance()
        psi_l = self.current_state.psi + self.drift_angle + angle_dt
        psi_c = self.current_state.psi + self.drift_angle
        psi_r = self.current_state.psi + self.drift_angle - angle_dt
        dx_l, dy_l = dist * np.cos(psi_l), dist * np.sin(psi_l)
        dx_c, dy_c = dist * np.cos(psi_c), dist * np.sin(psi_c)
        dx_r, dy_r = dist * np.cos(psi_r), dist * np.sin(psi_r)
        lines = [[(x_tail, y_tail), (x_tail + dx_l, y_tail + dy_l)],
                 [(x_tail, y_tail), (x_tail + dx_c, y_tail + dy_c)],
                 [(x_tail, y_tail), (x_tail + dx_r, y_tail + dy_r)]]
        scan_lanes = mc.LineCollection(lines, linestyles='--', linewidths=1, colors='b')
        axs.add_collection(scan_lanes)

        # plot current goal
        axs.scatter(self.current_goal.x, self.current_goal.y, marker="*", s=80, c="r")
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    @staticmethod
    def bind_to_range(x, lb, ub):
        return (lb if x < lb else ub) if x < lb or x > ub else x
