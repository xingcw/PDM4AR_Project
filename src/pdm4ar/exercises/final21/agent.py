import math
from matplotlib import collections as mc
import collections
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders

from pdm4ar.exercises.final21.dubins_RRT_starV2 import RRT
from pdm4ar.exercises.final21.mpc import MPCController

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
        self.current_state = None
        self.current_pos = None
        self.name = None
        self.path_x = None
        self.path_y = None
        self.psi = None
        self.counter = 0
        self.planing = True

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
        # SpacecraftState = sim_obs.players[self.name].state
        # print(f"vy: {SpacecraftState.vy}")
        # if np.abs(SpacecraftState.vx) < 0.01 and np.abs(SpacecraftState.vy) < 0.01:
        #     return SpacecraftCommands(acc_left=1, acc_right=-1)
        # else:
        #     return SpacecraftCommands(acc_left=SpacecraftState.vy-SpacecraftState.vx,
        #                               acc_right=-SpacecraftState.vx-SpacecraftState.vy)

        self.current_state = sim_obs.players[self.name].state
        state = [self.current_state.x, self.current_state.y, self.current_state.psi, self.current_state.vx, self.current_state.vy, self.current_state.dpsi]
        state = np.asarray(state)

        print(self.current_state)

        if self.planing:
            self.planning()
            self.planing = False

        self.counter += 1
        self.counter = self.counter % 10

        prediction_x = self.path_x[:1000]
        prediction_y = self.path_y[:1000]
        prediction_psi = self.psi[:1000]
        path = np.vstack((prediction_x.T,prediction_y.T,prediction_psi.T))
        path = path.T
        # acc_left, acc_right = self.controller(prediction_x, prediction_y, prediction_psi, -2, 5)
        command = self.mpccontroller(state, path)
        path = np.delete(path, [1])
        self.plot_state()
        return SpacecraftCommands(float(command[0]), float(command[1]))

    def controller(self, prediction_x, prediction_y, prediction_psi, k, miu):
        """
        Kiwan Version
        ---------------------------------
        This is the controller designed by a Gatech group, link: https://dcsl.gatech.edu/papers/tra02.pdf
        error-path tracking: link: https://pdf.sciencedirectassets.com/271599/1-s2.0-S0921889007X01635/1-s2.0-S0921889006001618/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEN3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDXI6hqKFLXWB1iJen5VxFtl3F4B4cKOplZ2sHXgY7yHwIgRToNwtAIWqeryHGY4JOdYvpkhnXGoKyAOQ47DOPyGJYqgwQI1v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDM5n9y%2F8rdWhR8JzxyrXA3hDTb2Q9Dqp%2FO17gHtbwmruePk%2FjOQuHC3vnLB6BBSo7czpknsukvnhApZE5S7IkcnoLYi088xXLyRcJ%2FX96wOS9eOpcbdHX0ee5b8GTATGexQwZ6zwJ8wQIoIi8j2eluAWwLRRongwPtnG6A7dzPnLVtE4%2Fo30gZdwxPDGXhiWHo%2FsV%2F7a1K%2B4IXtoBqvpT9SGxdKySyIhXwC3BR9%2BcSLpro9x4kbljQucbUDZFB%2Fx19p%2BjAus3sM74y%2F95%2BEOxS9htMFL6bJU75yiMoQ42wARSB8oB7mpJRcKBDCcYPdNLurvCB%2FFTu7ghAZZiOBd4L2Z7O7EkQMSyeUgvX7%2F6rGOcKb1jHvFYKtXbgX7xNvPQSBlsOWqxapxVVxLYn4Kf58C5%2BdTx23F89qTW2VM8scEwf7GGTt4bAYwn1g8u91xoY8uIyFUPHPuSbuFsmFpNOK2nrwystnpRpRbn8e1fzAMbJt8fFlIxmeFVwf1RDFymJAtRVLd6w4jfribcYXv64TYRG8o58MJF%2Bvylk8dDHLq%2BGJuQA59cYpRxOPSzJ9tDS1RW5Kwaqq2gfFUuXqEuFK8yxACZdzMSMCJq5X%2BJLNS%2B%2BmMJ3AILo6BJm4gAMEeCD6qe4G0kTC%2FprGOBjqlAam%2FZ0d095ptX8RYSPj1BA%2BYT2CLnFs2O6TK%2B0%2BSAUsIy9FhT4BB1pzrNdOojoF9j2x3wkMp2%2Bk4FqiUTD34HwQvz%2FHaJsEA0bM77KBg4%2BPVBAJybWMklYm69AAvgwIoD%2FDlFhF7BSes9aGtbuQ9Cq5z0qLOYm1SSdKA7EMFz11c8uTp5lyJjRQrBVT%2FA700gaXo1nRdjyrwlMgKh7P5DgN93kUv1g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211229T124844Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZJOP45YM%2F20211229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=618b510a2299ac5e32562e16c74a9c164b9b33fcef13314fa8d2e6489d3e4d1e&hash=b423d90a3f70705f4388c426a3410712e63b61990e2bd333ae58586e69d98daf&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0921889006001618&tid=spdf-f6faf625-1a07-49b3-ad34-6f422e744cb7&sid=c90d19493c1f824cd698ea6794d187af7f8agxrqb&type=client
        path_tracking: http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_ICRA_2008/Multi-Robot%20System%20with%20Kinematic%20or%20Dynamic%20Constraints/Formation%20path%20following%20control%20of%20unicycle-type%20mobile%20rob.pdf
        If you want to understand the algo, just go through the nice paper
        """
        x = self.current_state.x
        y = self.current_state.y
        psi = self.current_state.psi
        vx = self.current_state.vx
        vy = self.current_state.vy
        dpsi = self.current_state.dpsi

        x1 = (prediction_x - x) * math.cos(psi) + (prediction_y - y) * math.sin(psi)
        x2 = prediction_psi - psi
        x3 = (prediction_x - x) * math.sin(psi) - (prediction_y - y) * math.cos(psi)
        s = x3 - 0.5 * x1 * x2

        # this is the velocity of the 'wheels' in the future time stamp
        # u1 = -k * x1 + miu * s / (x1 ** 2 + x2 ** 2) * x2
        # u2 = -k * x2 - miu * s / (x1 ** 2 + x2 ** 2) * x1
        viu = (x1 ** 2 + x2 ** 2) ** 0.5
        if viu != 0:
            sat_2 = min(1, abs(s / viu)) * np.sign(s / viu) * x2 / viu
            sat_1 = min(1, abs(s / viu)) * np.sign(s / viu) * x1 / viu
        else:
            sat_2 = np.sign(s)
            sat_1 = np.sign(s)

        u1 = -k * x1 / (viu ** 2 + 1) + miu * sat_2
        u2 = -k * x2 / (viu ** 2 + 1) - miu * sat_1

        # this is the velocity of the 'wheels' at the current time stamp
        width = self.sg.w_half * 2
        v_left = vx - width / 2 * dpsi
        v_right = vx + width / 2 * dpsi

        # return the acceleration
        acc_left = (u1 - v_left) / 0.5
        acc_right = (u2 - v_right) / 0.5

        return acc_left, acc_right

    def mpccontroller(self, state, plan_seq):

        mpc_controller = MPCController(plan_seq, self.sg, state)
        command = mpc_controller.mpc_command(state, plan_seq)
        return command

    def planning(self):
        """
        Kiwan Version
        ---------------------------------
        This is how we plan the route for the rocket to follow
        """

        x_start = (self.current_state.x, self.current_state.y, self.current_state.psi)
        x_goal = (self.goal.goal.centroid.x, self.goal.goal.centroid.y, math.pi/2)
        rrt = RRT(self.static_obstacles)
        rrt.set_start(x_start)
        rrt.run(x_goal, nb_iteration=100)
        self.path_x, self.path_y, self.psi = rrt.get_final_path()

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

        # plot planned path
        axs.plot(self.path_x, self.path_y, '-k', linewidth=2)

        # plot velocities
        x_tail = self.current_state.x
        y_tail = self.current_state.y
        dx = np.abs(self.current_state.vx) * np.cos(self.current_state.psi)
        dy = np.abs(self.current_state.vx) * np.sin(self.current_state.psi)
        axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.3, length_includes_head=True, color="r")
        # axs.text(x_tail + 1, y_tail - .4, "$V_x$")

        v = np.linalg.norm(np.array([self.current_state.vx, self.current_state.vy]))
        dx = v * np.cos(self.current_state.psi + math.atan2(self.current_state.vy,self.current_state.vx))
        dy = v * np.sin(self.current_state.psi + math.atan2(self.current_state.vy,self.current_state.vx))
        axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.3, length_includes_head=True, color="k")
        # axs.text(x_tail + 1, y_tail - .4, "V")

        # dx = np.abs(self.current_state.vy) * np.cos(self.current_state.psi + np.pi / 2)
        # dy = np.abs(self.current_state.vy) * np.sin(self.current_state.psi + np.pi / 2)
        # axs.arrow(x_tail + 1, y_tail - .4, dx, dy, width=0.5, length_includes_head=True, color="k")
        # axs.text(x_tail + 1, y_tail - .4, "$V_y$")

        # plot scan lanes
        # angle_dt = 0.1
        # dist = self.get_safe_distance()
        # psi_l = self.current_state.psi + angle_dt
        # psi_c = self.current_state.psi
        # psi_r = self.current_state.psi - angle_dt
        # dx_l, dy_l = dist * np.cos(psi_l), dist * np.sin(psi_l)
        # dx_c, dy_c = dist * np.cos(psi_c), dist * np.sin(psi_c)
        # dx_r, dy_r = dist * np.cos(psi_r), dist * np.sin(psi_r)
        # lines = [[(x_tail, y_tail), (x_tail + dx_l, y_tail + dy_l)],
        #          [(x_tail, y_tail), (x_tail + dx_c, y_tail + dy_c)],
        #          [(x_tail, y_tail), (x_tail + dx_r, y_tail + dy_r)]]
        # scan_lanes = mc.LineCollection(lines, linestyles='--', linewidths=1, colors='b')
        # axs.add_collection(scan_lanes)

        # plot current goal
        axs.scatter(self.goal.goal.centroid.x, self.goal.goal.centroid.y, marker="*", s=80, c="r")
        plt.savefig("mygraph.png")
