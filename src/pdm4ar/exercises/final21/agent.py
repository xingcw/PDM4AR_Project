import random
from typing import Sequence, List
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
        self.debug = True
        self.planner = None
        self.current_state = None
        self.start_pos = None
        self.goal_pos = None
        self.waypoints = []
        self.dpoints = []
        self.stops = []
        self.visited_pts = []
        # TODO: get rid of time stamp
        self.t_step = 0
        self.name = None
        self.player = None
        # TODO: remove later
        self.C0 = None

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
        self.player = sim_obs.players[self.name]
        self.current_state = self.player.state
        self.visited_pts.append([self.current_state.x, self.current_state.y])
        if self.planner is None:
            self.start_pos = Node.from_state(self.current_state)
            self.goal_pos = Node([self.goal.goal.centroid.x, self.goal.goal.centroid.y])
            self.replan(self.start_pos, self.goal_pos)
            self.stops = self.get_stops()
            while self.stops is None:
                print(f"[replanning] cannot find available number of stops!")
                self.replan(self.start_pos, self.goal_pos)
                self.stops = self.get_stops()
            self.dpoints, self.C0 = self.fit_turning_curve(self.stops)

        self.t_step += 1
        dpoints = self.dpoints[self.t_step:]
        if len(dpoints) < 30:
            self.dpoints.append(self.dpoints[-1])
            dpoints.append(self.dpoints[-1])
        try:
            controller = MPCController(dpoints, self.sg, self.current_state.as_ndarray())
            commands = controller.mpc_command(self.current_state.as_ndarray(), dpoints).squeeze()
            commands = SpacecraftCommands(acc_left=commands[0], acc_right=commands[1])
        except:
            commands = SpacecraftCommands(acc_left=0, acc_right=0)
        print(commands)
        print(self.current_state)
        if self.debug:
            self.plot_state()
        return commands

    def get_stops(self, max_num_steps=8):
        start_point = Node.from_state(self.current_state)
        end_point = start_point
        origin_waypoints = self.waypoints
        stops = [start_point]

        while not end_point.equal(self.goal_pos) and len(stops) < max_num_steps:
            end_point = self.get_furthest_no_collision_waypoint(start_point)
            if end_point is None:
                return None
            stops.append(end_point)
            start_point = end_point

        self.waypoints = origin_waypoints
        return stops if end_point.equal(self.goal_pos) else None

    def get_furthest_no_collision_waypoint(self, current_pos=None, offset=3.0):
        if current_pos is None:
            current_pos = Node.from_state(self.current_state)
        if not self.planner.is_collision(current_pos, self.planner.s_goal, offset):
            return self.waypoints[-1]
        collision = np.array(
            [self.planner.is_collision(current_pos, waypoint, offset) for waypoint in self.waypoints])
        state = np.array([current_pos.x, current_pos.y])
        waypoints = np.array([[waypoint.x, waypoint.y] for waypoint in self.waypoints])
        distance = np.linalg.norm(state - waypoints, axis=1)
        no_collision_dist = np.multiply(distance, ~collision)
        idx = np.argmax(no_collision_dist)
        candidate_goal = self.waypoints[idx]

        if self.sampling_check_collision(candidate_goal, offset):
            new_candidate = self.resample_goal(candidate_goal, offset)
            if new_candidate is None:
                if not self.sampling_check_collision(candidate_goal.parent, offset) and \
                        not current_pos.equal(candidate_goal.parent):
                    print(f"[resample next goal] sampling not available, try its parent...")
                    candidate_goal = candidate_goal.parent
                    idx -= 1
                else:
                    return None
            else:
                print(f"[resample next goal] sampling... \n")
                candidate_goal = new_candidate

        self.waypoints = self.waypoints[idx:]
        return candidate_goal

    def sampling_check_collision(self, waypoint: Node, offset=3.0, radius=5.0):
        if waypoint.equal(self.goal_pos):
            return False
        sampling_angles = np.linspace(-np.pi, np.pi, 12)
        for aa in sampling_angles:
            x = waypoint.x + radius * np.cos(aa)
            y = waypoint.y + radius * np.sin(aa)
            check_point = Node([x, y])
            if self.planner.is_collision(waypoint, check_point, offset):
                return True
        return False

    def resample_goal(self, candidate: Node, offset=3.0, radius=5.0):
        current_pos = Node.from_state(self.current_state)
        dist, psi = current_pos.point_to(candidate)
        sampling_angles = np.linspace(psi - np.pi / 2, psi + np.pi / 2, 7)
        sampling_order = np.flipud(np.argsort(np.abs(sampling_angles - psi)))
        sampling_angles = sampling_angles[sampling_order]
        # get the safe boundary of the environment
        x_range, y_range = self.planner.get_safe_env_bound(offset)
        for aa in sampling_angles:
            x = candidate.x + radius * np.cos(aa)
            y = candidate.y + radius * np.sin(aa)
            new_candidate = Node([x, y])
            if not self.planner.is_collision(current_pos, new_candidate, offset) and \
                    not self.sampling_check_collision(new_candidate, radius) \
                    and new_candidate.is_bound_in_range(x_range, y_range):
                return new_candidate
        return None

    def get_safe_distance(self):
        v = np.linalg.norm([self.current_state.vx, self.current_state.vy])
        return v ** 2 / (2 * A_MAX)

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

    def fit_turning_curve(self, waypoints: List[Node], turning_dist=10):
        dpoints = []
        C0 = []

        def get_dpoints(start_pt, start_dt, end_dt, step, angle):
            start_dts = np.linspace(start_dt, end_dt, np.ceil((end_dt - start_dt) / step).astype(int), endpoint=False)
            dx, dy = start_dts * np.cos(angle), start_dts * np.sin(angle)
            ddpts = [Node([p[0] + start_pt.x, p[1] + start_pt.y]) for p in np.vstack([dx, dy]).transpose()]
            for p in ddpts:
                p.psi = angle
            return ddpts

        for start, mid, end in zip(waypoints[:-2], waypoints[1:-1], waypoints[2:]):
            dist_1, psi_1 = start.point_to(mid)
            dist_2, psi_2 = mid.point_to(end)
            turning_dist = min(turning_dist, dist_1, dist_2) / 2
            step_size = self.bind_to_range(max(dist_1, dist_2) / 50, 0.2, 1.0)

            # discretize start/2 -> mid
            start_dist = 0 if start.equal(waypoints[0]) else dist_1 / 2
            dpoints.extend(get_dpoints(start, start_dist, dist_1-turning_dist, step_size, psi_1))

            # discretize start/2 -> mid -> end/2
            # turning_radius is defined as positive when psi_1 < psi_2,
            # and the marching direction is along positive x- and y- axis
            if np.abs(psi_2 - psi_1) > 0.2:
                turning_radius = turning_dist * np.tan((np.pi - (psi_2 - psi_1)) / 2)
                long_side = turning_dist / np.cos((np.pi - (psi_2 - psi_1)) / 2)
                curve_origin = [mid.x - long_side * np.cos((np.pi - (psi_2 + psi_1)) / 2),
                                mid.y + long_side * np.sin((np.pi - (psi_2 + psi_1)) / 2)]
                C0.append(curve_origin)
                daa = 0.5 * step_size / turning_radius
                for aa in np.arange(0, psi_2 - psi_1, daa):
                    dpoint = Node([curve_origin[0] + turning_radius * np.sin(psi_1 + aa),
                                   curve_origin[1] - turning_radius * np.cos(psi_1 + aa)])
                    dpoint.psi = psi_1 + aa
                    dpoints.append(dpoint)
            else:
                dpoints.extend(get_dpoints(start, dist_1-turning_dist, dist_1, step_size, psi_1))
                dpoints.extend(get_dpoints(mid, 0, turning_dist, step_size, psi_2))

            # discretize mid -> end
            end_dist = dist_2 if end.equal(waypoints[-1]) else dist_2 / 2
            ddpoints = get_dpoints(mid, turning_dist, end_dist, step_size, psi_2)
            dpoints.extend(ddpoints)

        end_pos = self.goal_pos
        end_pos.psi = dpoints[-1].psi
        dpoints.append(end_pos)

        return dpoints, np.array(C0).reshape(-1, 2)

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
        axs.plot([x[0] for x in self.planner.path], [x[1] for x in self.planner.path], '-k', linewidth=1)

        # plot discretized path
        axs.scatter(self.C0[:, 0], self.C0[:, 1], marker="+", s=80, c="r")
        axs.scatter([dp.x for dp in self.dpoints], [dp.y for dp in self.dpoints], marker=".", s=2, c="r")

        # plot visited points
        visited_path = np.array(self.visited_pts).reshape(-1, 2)
        axs.scatter(visited_path[:, 0], visited_path[:, 1], marker=".", s=2, c="b")

        # plot velocities
        x_tail = self.current_state.x
        y_tail = self.current_state.y
        dx = np.abs(self.current_state.vx) * np.cos(self.current_state.psi)
        dy = np.abs(self.current_state.vx) * np.sin(self.current_state.psi)
        axs.arrow(x_tail, y_tail, dx, dy, width=0.3, length_includes_head=True, color="r")
        # axs.text(x_tail + 1, y_tail - .4, "$V_x$")

        v = np.linalg.norm(np.array([self.current_state.vx, self.current_state.vy]))
        drift_angle = np.arctan2(self.current_state.vy, self.current_state.vx)
        dx = v * np.cos(self.current_state.psi + drift_angle)
        dy = v * np.sin(self.current_state.psi + drift_angle)
        axs.arrow(x_tail, y_tail, dx, dy, width=0.3, length_includes_head=True, color="k")
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
        stops = np.array([[s.x, s.y] for s in self.stops])
        axs.scatter(stops[:, 0], stops[:, 1], marker="*", s=80, c="r")

        # plot players
        axs.plot(*self.player.occupancy.boundary.xy)

        # TODO: plot the open-loop and closed-loop trajectory

        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    @staticmethod
    def bind_to_range(x, lb, ub):
        return (lb if x < lb else ub) if x < lb or x > ub else x
