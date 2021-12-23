import matplotlib
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from RRT_star import RrtStar

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
    # plt.show()

    x_start = (x0.x, x0.y)
    x_goal = (goal.goal.centroid.x, goal.goal.centroid.y)

    rrt_star = RrtStar(x_start, x_goal, 10, 0.10, 20, 2000, dg_scenario)
    rrt_star.planning()
