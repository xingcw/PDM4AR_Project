import do_mpc
import numpy as np
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry

class MPC_controller(object):
    def __init__(self,plan_seq,SpacecraftGeometry,**kwargs):
        # either 'discrete' or 'continuous'
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        #define state
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        phi = model.set_variable(var_type='_x', var_name='phi', shape=(1, 1))
        vx = model.set_variable(var_type='_x', var_name='vx', shape=(1, 1))
        vy = model.set_variable(var_type='_x', var_name='vy', shape=(1, 1))
        dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(1, 1))

        #define input
        acc_l = model.set_variable(var_type='_u', var_name='acc_l')
        acc_r = model.set_variable(var_type='_u', var_name='acc_r')

        #define target as time varying parameter
        x_gt = model.set_variable(var_type='_tvp', var_name='x_gt', shape=(1, 1))
        y_gt = model.set_variable(var_type='_tvp', var_name='y_gt', shape=(1, 1))


        #setup model parameter
        w = SpacecraftGeometry.w_half*2
        lr = SpacecraftGeometry.lr
        lf = SpacecraftGeometry.lf


        #system dynamics
        model.set_rhs('x', vx * np.cos(phi) - vy * np.sin(phi))
        model.set_rhs('y', vx * np.sin(phi) + vy * np.cos(phi))
        model.set_rhs('phi' , dphi)
        model.set_rhs('vx', acc_l + acc_r + vy * dphi)
        model.set_rhs('vy', - vx * dphi)
        model.set_rhs('dphi' , 0.5 * w * (acc_r - acc_l))

        #finish model setup
        model.setup()
        self.model = model

        #setup MPC controller
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)
        self.n_horizon = setup_mpc['n_horizon']

        #objective setup
        mterm = (x-x_gt) ** 2 + (y-y_gt) ** 2
        lterm = (x-x_gt) ** 2 + (y-y_gt) ** 2

        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(
            acc_l=1e-2,
            acc_r=1e-2
        )

        #constraints
        # Lower bounds on states:
        mpc.bounds['lower', '_x', 'vx'] = -50
        mpc.bounds['lower', '_x', 'vy'] = -50
        mpc.bounds['lower', '_x', 'dphi'] = -2 * np.pi
        # Upper bounds on states
        mpc.bounds['upper', '_x', 'vx'] = 50
        mpc.bounds['upper', '_x', 'vy'] = 50
        mpc.bounds['upper', '_x', 'dphi'] = 2 * np.pi

        # Lower bounds on inputs:
        mpc.bounds['lower', '_u', 'acc_l'] = - 10
        mpc.bounds['lower', '_u', 'acc_r'] = - 10
        mpc.bounds['upper', '_u', 'acc_l'] = 10
        mpc.bounds['upper', '_u', 'acc_r'] = 10


        self.mpc = mpc
        self.update_planned_target(plan_seq)
        self.mpc.setup()

        self.mpc.x0 = np.zeros((6,1)).reshape(-1,1)
        self.mpc.set_initial_guess()

    def print_state(self):
        return  self.model.x.labels()

    def mpc_command(self,state,planned_seq):
        self.update_planned_target(planned_seq)
        return self.mpc.make_step(state)

    def update_planned_target(self,plan_sequence):
        tvp_template = self.mpc.get_tvp_template()
        def tvp_fun(t_now):
            for k in range(self.n_horizon + 1):
                tvp_template['_tvp', k, 'x_gt'] = plan_sequence[k,0]
                tvp_template['_tvp', k, 'y_gt'] = plan_sequence[k,1]
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun)


    def debug_in_simulation(self):
        pass

if __name__ == '__main__':
    import collections
    SpacecraftGeometry = collections.namedtuple('SpacecraftGeometry',['w_half','lr','lf'])
    _SpacecraftGeometry = SpacecraftGeometry(1,1,1)
    state = np.zeros((6,1)).reshape(-1,1) #curent state from observation
    plan_seq = np.ones((100, 2))  #target sequence from planner


    mpc_controller = MPC_controller(plan_seq,_SpacecraftGeometry)
    command = mpc_controller.mpc_command(state,plan_seq)
    print(command)
