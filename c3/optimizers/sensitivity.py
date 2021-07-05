"""Module for Sensitivity Analysis. This allows the sweeping of the goal
function in a given range of parameters to ascertain whether the dataset
being used is sensitive to changes in the parameters of interest"""

from copy import deepcopy
import os
from typing import Any, Dict, List, Tuple
from c3.optimizers.modellearning import ModelLearning
from c3.parametermap import ParameterMap
from c3.utils.utils import flatten


class Sensitivity(ModelLearning):
    """Class for Sensitivity Analysis, subclassed from Model Learning

    Parameters
    ----------
    sampling : str
        Name of the sampling method from library
    batch_sizes : Dict[str, int]
        Number of points to select from the dataset
    pmap : ParameterMap
        Model parameter map
    datafiles : Dict[str, str]
        The datafiles for each of the learning datasets
    state_labels : Dict[str, List[Any]]
        The labels for the excited states of the system
    sweep_map : List[List[List[str]]]
        Map of variables to be swept in exp_opt_map format
    sweep_bounds : List[List[int]]
        List of upper and lower bounds for each sweeping variable
    algorithm : str
        Name of the sweeping algorithm from the library
    estimator : str
        Name of estimator method from library
    estimator_list : List[str]
        List of different estimators to be used
    dir_path : str, optional
        Path to save sensitivity logs, by default None
    run_name : str, optional
        Name of the experiment run, by default None
    options : dict, optional
        Options for the sweeping algorithm, by default {}

    Raises
    ------
    NotImplementedError
        When trying to set the estimator or estimator_list
    """

    def __init__(
        self,
        sampling: str,
        batch_sizes: Dict[str, int],
        pmap: ParameterMap,
        datafiles: Dict[str, str],
        state_labels: Dict[str, List[Any]],
        sweep_map: List[List[Tuple[str]]],
        sweep_bounds: List[List[int]],
        algorithm: str,
        estimator: str = None,
        estimator_list: List[str] = None,
        dir_path: str = None,
        run_name: str = None,
        options={},
    ) -> None:

        super().__init__(
            sampling,
            batch_sizes,
            pmap,
            datafiles,
            dir_path,
            estimator,
            state_labels=state_labels,
            algorithm=algorithm,
            run_name=run_name,
            options=options,
        )
        if estimator_list:
            raise NotImplementedError(
                "C3:ERROR: Estimator Lists are currently not supported."
                "Only the standard logarithmic likelihood can be used at the moment."
                "Please remove this setting."
            )
        self.sweep_map = sweep_map  # variables to be swept
        self.pmap.opt_map = [
            sweep_map[0]
        ]  # set the opt_map to the first sweep variable
        self.sweep_bounds = sweep_bounds  # bounds for sweeping
        self.sweep_end: List[
            Dict[Any, Any]
        ] = list()  # list for storing the params and goals at the end of the sweep
        self.scaling = False  # interoperability with model learning which uses scaling
        self.logname = "sensitivity.log"  # shared log_setup requires logname
        self.run = self.sensitivity  # alias for legacy method

    def sensitivity(self):
        """
        Run the sensitivity analysis.
        """
        for ii in range(len(self.sweep_map)):
            self.pmap.opt_map = [self.sweep_map[ii]]
            self.options["bounds"] = [self.sweep_bounds[ii]]
            print(f"C3:STATUS:Sweeping {self.pmap.opt_map}: {self.sweep_bounds[ii]}")
            self.log_setup()
            self.start_log()
            print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
            x_init = self.pmap.get_parameters_scaled()
            try:
                self.algorithm(
                    x_init,
                    fun=self.fct_to_min,
                    fun_grad=self.fct_to_min_autograd,
                    grad_lookup=self.lookup_gradient,
                    options=self.options,
                )
            except KeyboardInterrupt:
                pass
            temp_param_name = "".join(flatten(self.pmap.opt_map))
            self.sweep_end.append({temp_param_name: deepcopy(self.optim_status)})
