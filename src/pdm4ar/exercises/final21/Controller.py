import do_mpc
import numpy as np
from typing import Sequence
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from dg_commons.sim.models.spacecraft import SpacecraftState
from pdm4ar.exercises.final21.RRT_star import Node


class MPCController(object):
    def __init__(self, path: Sequence[Node], sg: SpacecraftGeometry, x0: SpacecraftState):
        # either 'discrete' or 'continuous'
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # define state
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        psi = model.set_variable(var_type='_x', var_name='psi', shape=(1, 1))
        vx = model.set_variable(var_type='_x', var_name='vx', shape=(1, 1))
        vy = model.set_variable(var_type='_x', var_name='vy', shape=(1, 1))
        dpsi = model.set_variable(var_type='_x', var_name='dpsi', shape=(1, 1))

        # define input
        acc_l = model.set_variable(var_type='_u', var_name='acc_l')
        acc_r = model.set_variable(var_type='_u', var_name='acc_r')

        # define target as time varying parameter
        x_gt = model.set_variable(var_type='_tvp', var_name='x_gt', shape=(1, 1))
        y_gt = model.set_variable(var_type='_tvp', var_name='y_gt', shape=(1, 1))
        vy_gt = model.set_variable(var_type='_tvp', var_name='vy_gt', shape=(1, 1))
        psi_gt = model.set_variable(var_type='_tvp', var_name='psi_gt', shape=(1, 1))

        # system dynamics
        model.set_rhs('x', vx * np.cos(psi) - vy * np.sin(psi))
        model.set_rhs('y', vx * np.sin(psi) + vy * np.cos(psi))
        model.set_rhs('psi', dpsi)
        model.set_rhs('vx', acc_l + acc_r + vy * dpsi)
        model.set_rhs('vy', - vx * dpsi)
        model.set_rhs('dpsi', sg.w_half * sg.m / sg.Iz * (acc_r - acc_l))

        # finish model setup
        model.setup()
        self.model = model

        # setup MPC controller
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': 30,
            't_step': 0.1,
            'n_robust': 0,
            'store_full_solution': False,
        }
        mpc.set_param(**setup_mpc)
        self.n_horizon = setup_mpc['n_horizon']

        # objective setup
        # TODO: tuning Q, R weights
        mterm = 3 * (x - x_gt) ** 2 + 3 * (y - y_gt) ** 2 + 5 * (psi - psi_gt) ** 2 + 2 * (vy - vy_gt) ** 2
        lterm = 3 * (x - x_gt) ** 2 + 3 * (y - y_gt) ** 2 + 5 * (psi - psi_gt) ** 2 + 2 * (vy - vy_gt) ** 2

        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(
            acc_l=1e-2,
            acc_r=1e-2
        )

        # constraints
        # Bounds on states:
        mpc.bounds['lower', '_x', 'vx'] = -50
        mpc.bounds['lower', '_x', 'vy'] = -50
        mpc.bounds['lower', '_x', 'dpsi'] = -2 * np.pi
        mpc.bounds['upper', '_x', 'vx'] = 50
        mpc.bounds['upper', '_x', 'vy'] = 50
        mpc.bounds['upper', '_x', 'dpsi'] = 2 * np.pi

        # Bounds on inputs:
        mpc.bounds['lower', '_u', 'acc_l'] = - 10
        mpc.bounds['lower', '_u', 'acc_r'] = - 10
        mpc.bounds['upper', '_u', 'acc_l'] = 10
        mpc.bounds['upper', '_u', 'acc_r'] = 10

        self.mpc = mpc
        self.update_planned_target(path)
        self.mpc.setup()

        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def print_state(self):
        return self.model.x.labels()

    def mpc_command(self, state, planned_seq):
        self.update_planned_target(planned_seq)
        return self.mpc.make_step(state)

    def update_planned_target(self, plan_sequence):
        tvp_template = self.mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(self.n_horizon):
                tvp_template['_tvp', k, 'x_gt'] = plan_sequence[k].x
                tvp_template['_tvp', k, 'y_gt'] = plan_sequence[k].y
                tvp_template['_tvp', k, 'psi_gt'] = plan_sequence[k].psi
                tvp_template['_tvp', k, 'vy_gt'] = 0
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun)

    def debug_in_simulation(self):
        pass


if __name__ == '__main__':
    import collections

    SpacecraftGeometry = collections.namedtuple('SpacecraftGeometry', ['w_half', 'lr', 'lf'])
    _SpacecraftGeometry = SpacecraftGeometry(1, 1, 1)
    state = np.zeros((6, 1)).reshape(-1, 1)  # curent state from observation
    plan_seq = np.ones((100, 2))  # target sequence from planner

    mpc_controller = MPCController(plan_seq, _SpacecraftGeometry)
    command = mpc_controller.mpc_command(state, plan_seq)
    print(command)
