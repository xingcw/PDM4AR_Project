from typing import Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from pdm4ar.exercises.final21.RRT_star import RrtStar, Node
from pdm4ar.exercises.final21.Controller import MPCController

A_MAX = 10
matplotlib.use('TkAgg')


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
        self.current_state = None
        self.current_goal = None
        self.waypoints = []
        self.dpoints = []
        self.drift_angle = 0
        # TODO: get rid of time stamp
        self.t_step = 0
        self.name = None

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
        self.current_state = sim_obs.players[self.name].state
        start_pos = Node.from_state(self.current_state)
        goal_pos = Node([self.goal.goal.centroid.x, self.goal.goal.centroid.y])
        self.drift_angle = np.arctan2(self.current_state.vy, self.current_state.vx)
        if self.planner is None:
            self.replan(start_pos, goal_pos)
            self.dpoints = self.discretize()
        # self.current_goal = self.waypoints[0]
        # if self.reach_goal():
        #     self.update_waypoints()
        # self.replan(start_pos, goal_pos)
        self.t_step += 1
        dpoints = self.dpoints[self.t_step:]
        controller = MPCController(dpoints, self.sg, self.current_state.as_ndarray())
        commands = controller.mpc_command(self.current_state.as_ndarray(), dpoints).squeeze()
        commands = SpacecraftCommands(acc_left=commands[0], acc_right=commands[1])
        print(commands)
        self.plot_state()
        return commands

    def get_safe_distance(self):
        v = np.linalg.norm([self.current_state.vx, self.current_state.vy])
        return v ** 2 / (2 * A_MAX)

    def reach_goal(self, thresh=3.0):
        if np.abs(self.current_state.x - self.current_goal.x) < thresh and \
                np.abs(self.current_state.y - self.current_goal.y) < thresh:
            return True
        else:
            return False

    def update_waypoints(self):
        self.waypoints.remove(self.current_goal)

    def replan(self, start, end):
        self.planner = RrtStar(start, end, self.static_obstacles, safe_offset=3.0)
        self.planner.planning()
        self.waypoints = []

        for node in reversed(self.planner.path):
            waypoint = Node(node)
            self.waypoints.append(waypoint)
        self.waypoints[0].parent = self.waypoints[0]
        self.waypoints[-1].child = self.waypoints[-1]

        for i, waypoint in enumerate(self.waypoints):
            if waypoint.parent is None:
                waypoint.parent = self.waypoints[i - 1]
            if waypoint.child is None:
                waypoint.child = self.waypoints[i + 1]
        for waypoint in self.waypoints:
            dist, angle = waypoint.point_to(waypoint.child)
            waypoint.psi = angle

    def discretize(self, horizon=20):
        # TODO: resolve the discretization around the terminal state
        dpoints = []
        for start, end in zip(self.waypoints[:-1], self.waypoints[1:]):
            dist, aa = start.point_to(end)
            dists = np.linspace(0, dist, horizon, endpoint=False)
            dxs, dys = dists * np.cos(aa), dists * np.sin(aa)
            ddpoints = [Node([p[0] + start.x, p[1] + start.y]) for p in np.vstack([dxs, dys]).transpose()]
            for p in ddpoints:
                p.psi = aa
            dpoints.extend(ddpoints)
        return dpoints

    def plot_state(self):
        # axs = plt.gca()
        _, axs = plt.subplots()
        shapely_viz = ShapelyViz(axs)

        # plot original obstacles
        for s_obstacle in self.static_obstacles:
            shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
        shapely_viz.add_shape(self.goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
        axs = shapely_viz.ax
        axs.autoscale()
        axs.set_aspect("equal")

        # plot safe boundary of obstacles
        for safe_s_obstacle in self.planner.safe_s_obstacles[self.planner.offset]:
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
        angle_dt = 0.1
        dist = self.get_safe_distance()
        psi_l = self.current_state.psi + angle_dt
        psi_c = self.current_state.psi
        psi_r = self.current_state.psi - angle_dt
        dx_l, dy_l = dist * np.cos(psi_l), dist * np.sin(psi_l)
        dx_c, dy_c = dist * np.cos(psi_c), dist * np.sin(psi_c)
        dx_r, dy_r = dist * np.cos(psi_r), dist * np.sin(psi_r)
        lines = [[(x_tail, y_tail), (x_tail + dx_l, y_tail + dy_l)],
                 [(x_tail, y_tail), (x_tail + dx_c, y_tail + dy_c)],
                 [(x_tail, y_tail), (x_tail + dx_r, y_tail + dy_r)]]
        scan_lanes = mc.LineCollection(lines, linestyles='--', linewidths=1, colors='b')
        axs.add_collection(scan_lanes)

        # plot current goal
        # axs.scatter(self.current_goal.x, self.current_goal.y, marker="*", s=80, c="r")

        # TODO: plot the open-loop and closed-loop trajectory

        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    @staticmethod
    def bind_to_range(x, lb, ub):
        return (lb if x < lb else ub) if x < lb or x > ub else x
