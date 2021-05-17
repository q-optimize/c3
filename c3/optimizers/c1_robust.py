import time
import hjson
import tensorflow as tf
from c3.optimizers.c1 import C1
from c3.utils.utils import jsonify_list
import numpy as np


class C1_robust(C1):
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
    pmap : ParameterMap
        Identifiers for the parameter vector
    callback_fids : list of callable
        Additional fidelity function to be evaluated and stored for reference
    algorithm : callable
        From the algorithm library
        Save plots of control signals
    store_unitaries : boolean
        Store propagators as text and pickle
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(self, noise_map, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_map = noise_map

    def goal_run_with_grad(self, current_params):
        goals = []
        goals_float = []
        grads = []
        evaluation = int(self.evaluation)
        for noise_vals, noise_map in self.noise_map:
            orig_val = np.array(self.exp.pmap.get_parameters([noise_map]))
            for noise_val in noise_vals:
                self.exp.pmap.set_parameters([noise_val], [noise_map])
                self.evaluation = evaluation
                with tf.GradientTape() as t:
                    t.watch(current_params)
                    goal = self.goal_run(current_params)
                grad = t.gradient(goal, current_params)
                goals.append(goal)
                goals_float.append(float(goal))
                grads.append(grad)
            self.exp.pmap.set_parameters(orig_val, [noise_map])

        self.optim_status["goals_individual"] = [float(goal) for goal in goals]
        self.optim_status["goal_std"] = float(tf.math.reduce_std(goals))
        self.optim_status["gradient_std"] = (
            tf.math.reduce_std(grads, axis=0).numpy().tolist()
        )
        self.optim_status["goal"] = float(tf.reduce_mean(goals, axis=0))
        self.optim_status["time"] = time.asctime()
        return tf.reduce_mean(goals, axis=0), tf.reduce_mean(grads, axis=0)

    def start_log(self):
        """
        Initialize the log with current time.

        """
        super().start_log()
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("Robust values ")
            print(len(self.noise_map))
            logfile.write(hjson.dumps(jsonify_list(self.noise_map)))
            logfile.write("\n")
            logfile.flush()
