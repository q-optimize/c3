"""Object that deals with the model learning."""


class C3(Optimizer):
    """Object that deals with the model learning."""

    def __init__(self):
        """Initiliase."""
        pass

    def learn_model(
        self,
        exp,
        sim,
        eval_func,
        fom,
        callback_foms=[],
        opt_name='learn_model',
        settings={}
    ):
        # TODO allow for specific data from optimizer to be used for learning
        x0 = exp.get_parameters(self.opt_map, scaled=True)
        self.exp = exp
        self.sim = sim
        self.eval_func = eval_func
        self.fom = fom
        self.callback_foms = callback_foms
        self.opt_name = opt_name
        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        self.optim_status = {}
        self.evaluation = 0

        with open(self.logfile_name, 'a') as self.logfile:
            start_time = time.time()
            start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
            self.logfile.write("Starting optimization at ")
            self.logfile.write(start_time_str)
            self.logfile.write("Optimization parameters:\n\n")
            self.logfile.write(json.dumps(self.opt_map))
            self.logfile.write("\n")
            # TODO put optmizer specific code here
            if self.algorithm == 'cmaes':
                x_best = self.cmaes(
                    x0,
                    lambda x: self.goal_run_n(tf.constant(x)),
                    settings
                )

            elif self.algorithm == 'lbfgs':
                x_best = self.lbfgs(
                    x0,
                    lambda x: self.goal_run_n_with_grad(tf.constant(x)),
                    options=settings
                )

            elif self.algorithm == 'oneplusone':
                x_best = self.oneplusone(
                    tf.constant(x0),
                    self.goal_run_n,
                    options=settings
                )

            elif self.algorithm == 'keras-SDG':
                vars = tf.Variable(x0)
                self.keras_vars = vars
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
                optimizer.minimize(self.goal_run_n_keras, var_list=[vars])
                x_best = vars.numpy()

            elif self.algorithm == 'keras-Adam':
                vars = tf.Variable(x0)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
                optimizer.minimize(self.goal_run_n(vars), var_list=[vars])
                x_best = vars.numpy()

            else:
                raise Exception(
                    "I don't know the selected optimization algorithm."
                )
            self.exp.set_parameters(
                x_best, self.opt_map, scaled=True
            )
            end_time = time.time()
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime: {end_time-start_time}\n\n"
            )
            self.logfile.flush()
