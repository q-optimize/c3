"""Object that deals with the open loop optimal control."""

import os
import time
import json
import tensorflow as tf
import c3.utils.display as display
from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup


class C1(Optimizer):
    """
    Object that deals with the open loop optimal control.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    fid_func : callable
        infidelity function to be minimized
    fid_subspace : list
        Indeces identifying the subspace to be compared
    gateset_opt_map : list
        Hierarchical identifiers for the parameter vector
    opt_gates : list
        List of identifiers of gate to be optimized, a subset of the full gateset
    callback_fids : list of callable
        Additional fidelity function to be evaluated and stored for reference
    algorithm : callable
        From the algorithm library
    plot_dynamics : boolean
        Save plots of time-resolved dynamics in dir_path
    plot_pulses : boolean
        Save plots of control signals
    store_unitaries : boolean
        Store propagators as text and pickle
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(
        self,
        dir_path,
        fid_func,
        fid_subspace,
        gateset_opt_map,
        opt_gates,
        callback_fids=[],
        algorithm=None,
        plot_dynamics=False,
        plot_pulses=False,
        store_unitaries=False,
        options={},
        run_name=None
    ):
        super().__init__(
            algorithm=algorithm,
            plot_dynamics=plot_dynamics,
            plot_pulses=plot_pulses,
            store_unitaries=store_unitaries
            )
        self.opt_map = gateset_opt_map
        self.opt_gates = opt_gates
        self.fid_func = fid_func
        self.fid_subspace = fid_subspace
        self.callback_fids = callback_fids
        self.options = options
        self.log_setup(dir_path, run_name)

    def log_setup(self, dir_path, run_name):
        """
        Create the folders to store data.

        Parameters
        ----------
        dir_path : str
            Filepath
        run_name : str
            User specified name for the run

        """
        self.dir_path = os.path.abspath(dir_path)
        if run_name is None:
            run_name = (
                'c1_' + self.fid_func.__name__ + '_' + self.algorithm.__name__
            )
        self.logdir = log_setup(self.dir_path, run_name)
        self.logname = 'open_loop.log'

    def load_best(self, init_point):
        """
        Load a previous parameter point to start the optimization from.

        Parameters
        ----------
        init_point : str
            File location of the initial point

        """
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_gateset_opt_map = [
                [tuple(par) for par in set]
                for set in json.loads(best[0])
            ]
            init_p = json.loads(best[1])['params']
            self.exp.gateset.set_parameters(init_p, best_gateset_opt_map)

    def adjust_exp(self, adjust_exp):
        """
        Load values for model parameters from file.

        Parameters
        ----------
        adjust_exp : str
            File location for model parameters

        """
        with open(adjust_exp) as file:
            best = file.readlines()
            best_exp_opt_map = [tuple(a) for a in json.loads(best[0])]
            p = json.loads(best[1])['params']
            self.exp.set_parameters(p, best_exp_opt_map)

    def optimize_controls(self):
        """
        Apply a search algorithm to your gateset given a fidelity function.
        """
        self.start_log()
        self.exp.set_enable_dynamics_plots(self.plot_dynamics, self.logdir)
        self.exp.set_enable_pules_plots(self.plot_pulses, self.logdir)
        self.exp.set_enable_store_unitaries(self.store_unitaries, self.logdir)
        self.exp.set_opt_gates(self.opt_gates)
        self.nice_print = self.exp.gateset.print_parameters
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        index = []
        for name in self.fid_subspace:
            index.append(self.exp.model.names.index(name))
        self.index = index
        x0 = self.exp.gateset.get_parameters(self.opt_map, scaled=True)
        try:
            self.algorithm(
                x0,
                fun=self.fct_to_min,
                fun_grad=self.fct_to_min_autograd,
                grad_lookup=self.lookup_gradient,
                options=self.options
            )
        except KeyboardInterrupt:
            pass
        with open(self.logdir + 'best_point_' + self.logname, 'r') as file:
            best_params = json.loads(file.readlines()[1])['params']
        self.exp.gateset.set_parameters(best_params, self.opt_map)
        self.end_log()

    def goal_run(self, current_params):
        """
        Evaluate the goal function for current parameters.

        Parameters
        ----------
        current_params : tf.Tensor
            Vector representing the current parameter values.

        Returns
        -------
        tf.float64
            Value of the goal function
        """
        self.exp.gateset.set_parameters(
            current_params,
            self.opt_map,
            scaled=True
        )
        dims = self.exp.model.dims
        U_dict = self.exp.get_gates()
        goal = self.fid_func(U_dict, self.index, dims, self.evaluation + 1)
        goal_numpy = float(goal.numpy())
        try:
            display.plot_C1(self.logdir)
        except TypeError:
            pass

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write(f"\nEvaluation {self.evaluation + 1} returned:\n")
            logfile.write(
                "goal: {}: {}\n".format(self.fid_func.__name__, goal_numpy)
            )
            for cal in self.callback_fids:
                val = cal(
                    U_dict, self.index, dims, self.logdir, self.evaluation + 1
                )
                if isinstance(val, tf.Tensor):
                    val = float(val.numpy())
                logfile.write("{}: {}\n".format(cal.__name__, val))
                self.optim_status[cal.__name__] = val
            logfile.flush()

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.gateset.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        self.optim_status['time'] = time.asctime()
        self.evaluation += 1
        return goal
