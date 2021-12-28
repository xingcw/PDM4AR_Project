import matplotlib
import math
import numpy
import numpy as np
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from RRT_star import RrtStar
from dubins_RRT_starV2 import RRT

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    dg_scenario, goal, x0 = get_dgscenario()
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in dg_scenario.static_obstacles.values():
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_facecolor('k')
    ax.set_aspect("equal")
    # x = [1,20,30]
    # y = [1,20,30]
    # plt.plot(x,y)
    # plt.show()

    # x_start = (x0.x, x0.y)
    # x_goal = (goal.goal.centroid.x, goal.goal.centroid.y)
    #
    # rrt_star = RrtStar(x_start, x_goal, 10, 0.10, 20, 2000, dg_scenario)
    # rrt_star.planning()

    '''For Kiwan testing, dubins RRT star algo'''
    rrt = RRT(dg_scenario)

    # We select two random points in the free space as a start and final node
    start = (x0.x, x0.y, math.pi/2)
    end = (goal.goal.centroid.x, goal.goal.centroid.y, math.pi/2)

    # We initialize an empty tree
    rrt.set_start(start)

    # We run 100 iterations of growth
    rrt.run(end, nb_iteration=2000)
    rrt.plot(file_name='kiwan', nodes=True)
