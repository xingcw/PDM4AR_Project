from typing import Sequence, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from shapely.geometry import LineString
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
        self.npc = []
        self.dvos = None
        self.dynamic = False
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

        if len(sim_obs.players.keys()) > 1:
            self.dynamic = True
            all_names = list(sim_obs.players.keys())
            all_names.remove(self.name)
            self.npc = []
            for npc in all_names:
                self.npc.append(sim_obs.players[npc])
            self.dvos = self.get_dynamic_velocity_obstacle()

        self.player = sim_obs.players[self.name]
        self.current_state = self.player.state
        self.visited_pts.append([self.current_state.x, self.current_state.y])

        if self.planner is None or self.is_path_in_collision():
            self.start_pos = Node.from_state(self.current_state)
            self.goal_pos = Node([self.goal.goal.centroid.x, self.goal.goal.centroid.y])
            self.replan(self.start_pos, self.goal_pos)
            self.stops = self.get_stops()
            while self.stops is None:
                print(f"[replanning] cannot find available stops!")
                self.replan(self.start_pos, self.goal_pos)
                self.stops = self.get_stops()
            self.dpoints, self.C0 = self.fit_turning_curve(self.stops)
        elif self.dynamic:
            self.planner.update_npc(self.dvos)

        self.t_step += 1
        dpoints = self.dpoints[self.t_step:]

        horizon = 40
        # add terminate state to the end of the path when the horizon is not reached
        if len(dpoints) < horizon:
            self.dpoints.append(self.dpoints[-1])
            dpoints.append(self.dpoints[-1])
        try:
            controller = MPCController(dpoints, self.sg, self.current_state.as_ndarray(), horizon)
            commands = controller.mpc_command(self.current_state.as_ndarray(), dpoints).squeeze()
            commands = SpacecraftCommands(acc_left=commands[0], acc_right=commands[1])
        except:
            commands = SpacecraftCommands(acc_left=0, acc_right=0)

        print("-"*20, f"Command at Time Step {self.t_step}", "-"*20)
        print(commands)
        print("-"*20, "Current State", "-"*20)
        print(self.current_state)

        if self.debug:
            self.plot_state()
        return commands

    def get_dynamic_velocity_obstacle(self):
        """
        Create a new obstacle for collision checks, which uses the velocity as the length,
        and the circumscribed circle as the base object.
        refer to shapely for more information: https://shapely.readthedocs.io/en/stable/manual.html#polygons
        :return: dynamic obstacles
        """
        dvos = []
        for npc in self.npc:
            minx, miny, maxx, maxy = npc.occupancy.bounds
            radius = np.linalg.norm([maxx-minx, maxy-miny]) / 2
            v = np.linalg.norm([npc.state.vx, npc.state.vy])
            aa = np.arctan2(npc.state.vy, npc.state.vx) + npc.state.psi
            dx, dy = v * np.cos(aa), v * np.sin(aa)
            start = [npc.state.x, npc.state.y]
            end = [2*dx + npc.state.x, 2*dy + npc.state.y]
            path = LineString([start, end])
            obs = path.buffer(distance=radius)
            dvos.append(obs)
        return dvos

    def is_path_in_collision(self, offset=3.0):
        """
        Get True if the simplified path is in collision with the dynamic obstacles.
        todo: remove the passed stops along the path
        :param offset: for collision checks
        :return:
        """
        if self.planner is None:
            return False
        for start, end in zip(self.stops[:-1], self.stops[1:]):
            if self.planner.is_collision(start, end, offset):
                return True
        return False

    def get_stops(self):
        """
        Get the simplified path with minimum number of stops.
        :return: list of stops if available, else None.
        """
        start_point = Node.from_state(self.current_state)
        end_point = start_point
        origin_waypoints = self.waypoints
        stops = [start_point]

        while not end_point.equal(self.goal_pos):
            end_point = self.get_furthest_no_collision_waypoint(start_point)
            if end_point is None:
                return None
            stops.append(end_point)
            start_point = end_point

        self.waypoints = origin_waypoints
        return stops if end_point.equal(self.goal_pos) else None

    def get_furthest_no_collision_waypoint(self, current_pos=None, offset=3.0, search_radius=30):
        """
        Used for getting the simplified path.
        Get the furthest node/waypoint from the current node/waypoint without collision with the environment.
        :param search_radius: max distance from current pose for searching waypoints.
        :param current_pos: the current node.
        :param offset: the safe offset defined for collision checks in this function. (nothing to do with
         the safe offset defined for the planner)
        :return:
        """
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
        valid_dist = no_collision_dist <= search_radius
        valid_no_collision_dist = np.multiply(no_collision_dist, valid_dist)
        idx = np.argmax(valid_no_collision_dist)

        # if the node cannot find a neighbor, skip it
        candidate_goal = self.waypoints[1] if idx == 0 else self.waypoints[idx]

        # check if the neighbor is safe enough, resample around it if not
        if self.sampling_check_collision(candidate_goal, offset):
            new_candidate = self.resample_goal(candidate_goal, current_pos, offset)
            if new_candidate is None:
                if not self.sampling_check_collision(candidate_goal.parent, offset) and \
                        not current_pos.equal(candidate_goal.parent):
                    print(f"[resample next goal] sampling fails, try its parent...")
                    candidate_goal = candidate_goal.parent
                    idx -= 1
                else:
                    return None
            else:
                print(f"[resample next goal] sampling... \n")
                candidate_goal = new_candidate

        # remove all the parents of the current node
        self.waypoints = self.waypoints[1:] if idx == 0 else self.waypoints[idx:]
        return candidate_goal

    def sampling_check_collision(self, waypoint: Node, offset=3.0, radius=5.0):
        """
        Check if a node/waypoint is safe enough by sampling points around it and check collisions.
        :param waypoint: the waypoint/node to be checked.
        :param offset: the safe offset defined for the collision checks.
        :param radius: the radius used for circle sampling.
        :return: True if not safe, else False.
        """
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

    def resample_goal(self, candidate: Node, current_pos, offset=3.0, radius=5.0):
        """
        Resample a new goal around the current goal which is not safe enough.
        :param current_pos: current node.
        :param candidate: target goal to be resampled.
        :param offset: safe offset for collision checks.
        :param radius: circle sampling radius.
        :return:
        """
        if current_pos is None:
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

    def replan(self, start, end):
        self.planner = RrtStar(start, end, self.static_obstacles, self.dvos, safe_offset=3.0, step_len=30)
        self.planner.planning()
        self.waypoints = []

        for node in reversed(self.planner.path):
            waypoint = Node(node)
            self.waypoints.append(waypoint)
        self.waypoints[0].parent = self.waypoints[0]

        # the planner takes the nearest neighbor as the parent of the goal if it fails
        # so check it and replan from the last waypoint before reaching the goal if in collision
        if self.planner.is_collision(self.waypoints[-2], self.waypoints[-1], offset=3.0):
            # use a smaller step length for better paths
            goal_planner = RrtStar(self.waypoints[-2], self.waypoints[-1], self.static_obstacles,
                                   self.dvos, safe_offset=3.0, step_len=10)
            goal_planner.planning()
            # replanning util a solution is found
            while goal_planner.is_collision(Node(goal_planner.path[-2]), Node(goal_planner.path[-1]), offset=3.0):
                goal_planner.planning()
            self.waypoints.pop(-1)
            for node in reversed(goal_planner.path):
                waypoint = Node(node)
                self.waypoints.append(waypoint)

        self.waypoints[-1].child = self.waypoints[-1]

        for i, waypoint in enumerate(self.waypoints):
            if waypoint.parent is None:
                waypoint.parent = self.waypoints[i - 1]
            if waypoint.child is None:
                waypoint.child = self.waypoints[i + 1]
        for waypoint in self.waypoints:
            dist, angle = waypoint.point_to(waypoint.child)
            waypoint.psi = angle

    def fit_turning_curve(self, waypoints: List[Node], turning_dist=20):
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
                daa = 0.3 * step_size / turning_radius
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
        if self.dynamic:
            for npc in self.npc:
                shapely_viz.add_shape(npc.occupancy)
        axs = shapely_viz.ax
        axs.autoscale()
        axs.set_aspect("equal")

        # plot safe boundary of obstacles
        for safe_s_obstacle in self.planner.safe_obstacles[self.planner.offset]:
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

        v = np.linalg.norm(np.array([self.current_state.vx, self.current_state.vy]))
        drift_angle = np.arctan2(self.current_state.vy, self.current_state.vx)
        dx = v * np.cos(self.current_state.psi + drift_angle)
        dy = v * np.sin(self.current_state.psi + drift_angle)
        axs.arrow(x_tail, y_tail, dx, dy, width=0.3, length_includes_head=True, color="k")

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
