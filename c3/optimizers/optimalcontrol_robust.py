import time
import hjson
import tensorflow as tf
from c3.optimizers.optimalcontrol import OptimalControl
from c3.c3objs import hjson_encode
import numpy as np


class OptimalControlRobust(OptimalControl):
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

    def noise_instance(self, current_params, noise_val, noise_map, evaluation):
        self.exp.pmap.set_parameters([noise_val], [noise_map])
        self.evaluation = evaluation
        with tf.GradientTape() as t:
            t.watch(current_params)
            goal = self.goal_run(current_params)
        grad = t.gradient(goal, current_params)
        return goal, grad

    def goal_run_with_grad(self, current_params):
        goals = []
        goals_float = []
        grads = []
        evaluation = int(self.evaluation)
        for noise_vals, noise_map in self.noise_map:
            orig_val = np.array(self.exp.pmap.get_parameters([noise_map]))
            for noise_val in noise_vals:
                goal, grad = self.noise_instance(
                    current_params, noise_val, noise_map, evaluation
                )
                goals.append(goal)
                goals_float.append(float(goal))
                grads.append(grad)
            self.exp.pmap.set_parameters(orig_val, [noise_map])

        self.optim_status["goals_individual"] = [float(goal) for goal in goals]
        self.optim_status["goal_std"] = float(np.std(goals))
        self.optim_status["gradient_std"] = np.std(grads, axis=0).tolist()
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
            logfile.write(hjson.dumps(self.noise_map, default=hjson_encode))
            logfile.write("\n")
            logfile.flush()

    # Temporary fallback non @tf.function implementation for robust control
    def goal_run(self, current_params: tf.Tensor) -> tf.float64:
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
        self.pmap.set_parameters_scaled(current_params)
        dims = self.pmap.model.dims
        propagators = self.exp.compute_propagators()

        goal = self.fid_func(
            propagators=propagators,
            instructions=self.pmap.instructions,
            index=self.index,
            dims=dims,
            n_eval=self.evaluation + 1,
            **self.fid_func_kwargs,
        )
        return goal
