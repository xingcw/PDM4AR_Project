from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from pdm4ar.exercises.final21.RRT_star import RrtStar

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
        self.current_state = None
        self.name = None
        self.GO = True
        self.ROT = True
        self.max_vx = 0

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
        if self.current_goal is None:
            if np.abs(self.current_state.vx) > 0.1 or np.abs(self.current_state.vy) > 0.1:
                a_l = self.current_state.vy - self.current_state.vx
                a_r = - self.current_state.vy - self.current_state.vx
                return SpacecraftCommands(acc_left=a_l, acc_right=a_r)
            else:
                self.replan()
                self.current_goal = self.get_closest_waypoint()

        if self.reach_goal() and not self.ready_togo():
            self.GO = False
            self.ROT = True
            a = self.current_state.psi - self.current_goal.psi + 3 * self.current_state.dpsi
            command = SpacecraftCommands(acc_left=a, acc_right=-a)
            print(f"[reach goal and rotate] target psi: {self.current_goal.psi}")
        elif self.reach_goal() and self.ready_togo():
            self.GO = True
            self.ROT = False
            self.update_current_goal()
            print(f"[ready to go] next goal: 'x: {self.current_goal.x}, y: {self.current_goal.y}'")
        elif not self.reach_goal() and self.GO:
            delta_s = self.current_goal.point_to(self.current_state)
            dsa = np.linalg.norm(delta_s) if delta_s[0] < 0 else -np.linalg.norm(delta_s)
            dpsia = self.current_state.psi - self.current_goal.psi + 3 * self.current_state.dpsi
            al = self.bind_to_range(dsa - 2 * self.current_state.vx + 3 * dpsia, -A_MAX, A_MAX)
            ar = self.bind_to_range(dsa - 2 * self.current_state.vx - 3 * dpsia, -A_MAX, A_MAX)
            command = SpacecraftCommands(acc_left=al, acc_right=ar)
            print(f"[head to goal] goal: 'x: {self.current_goal.x}, y: {self.current_goal.y}' ")

        self.plot_state()
        print(self.current_state)
        print(command)

        return command

    def update_current_goal(self):
        path = self.current_goal.point_to(self.current_goal.child)
        self.max_vx = np.sqrt(A_MAX * np.linalg.norm(path))
        self.current_goal = self.current_goal.child

    def reach_goal(self, thresh=0.3):
        if np.abs(self.current_state.x - self.current_goal.x) < thresh and \
                np.abs(self.current_state.y - self.current_goal.y) < thresh and \
                np.abs(self.current_state.vx) < 0.1 and np.abs(self.current_state.vy) < 0.1:
            return True
        else:
            return False

    def ready_togo(self, thresh=0.1):
        if np.abs(self.current_state.psi - self.current_goal.psi) < thresh and np.abs(self.current_state.dpsi) < thresh:
            return True
        else:
            return False

    def get_closest_waypoint(self):
        state = np.array([self.current_state.x, self.current_state.y])
        waypoints = np.array([[waypoint.x, waypoint.y] for waypoint in self.waypoints])
        idx = np.argmin(waypoints - state)
        return self.waypoints[idx]

    def go_straight(self, accelerate):
        return SpacecraftCommands(acc_left=accelerate, acc_right=accelerate)

    def self_rotate(self, accelerate):
        return SpacecraftCommands(acc_left=accelerate, acc_right=-accelerate)

    def replan(self):
        dg_scenario, _, _ = get_dgscenario()
        x_start = (self.current_state.x, self.current_state.y)
        x_goal = (self.goal.goal.centroid.x, self.goal.goal.centroid.y)
        self.planner = RrtStar(x_start, x_goal, 10, 0.10, 20, 2000, dg_scenario)
        self.planner.planning()

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
        fig, axs = plt.subplots()
        axs.plot([x[0] for x in self.planner.path], [x[1] for x in self.planner.path], '-k', linewidth=2)

        x_tail = self.current_state.x
        y_tail = self.current_state.y
        dx = 10 * np.cos(self.current_state.psi)
        dy = 10 * np.sin(self.current_state.psi)
        axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.5, length_includes_head=True, color="C2")
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    @staticmethod
    def bind_to_range(x, lb, ub):
        return (lb if x < lb else ub) if x < lb or x > ub else x
