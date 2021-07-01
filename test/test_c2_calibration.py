from typing import List, Tuple
import tempfile
import hjson
import pytest
import numpy as np
import os

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.libraries import algorithms
from c3.main import run_cfg
from c3.optimizers.c2 import C2
from c3.parametermap import ParameterMap
from c3.utils.qt_utils import single_length_RB

OPT_CONFIG_FILE_NAME = "test/c2.cfg"
LOGDIR = os.path.join(tempfile.gettempdir(), "c2_logs")

# constants for mock_ORBIT
RB_NUMBER = 2  # number of orbit sequences
RB_LENGTH = 5  # length of each sequence
TARGET = 0  # target state for the orbit run
RESULT_VAL = 0.1  # averaged readout of the ORBIT
RESULTS_STD = 0.0  # std for the readouts
SHOTS = 10000  # number of shots for averaging


def mock_ORBIT(
    params: List[Quantity],
) -> Tuple[float, List[float], List[float], List[List[str]], List[int]]:
    """A function to mock ORBIT behaviour but return fixed non-changing values for
    the goal and results

    Parameters
    ----------
    params : List[Quantity]
        List of c3 Quantity style parameters that the black-box optimizer is working on

    Returns
    -------
    Tuple[float, List[float], List[float], List[List[str]], List[int]]
        goal : Result of goal function calculation, eg, mean of all results
        results : Averaged Readout of individial ORBIT sequences
        results_std : Standard deviation for each averaged readout
        seqs : ORBIT sequence that was run, eg, ["rx90p", "ry90p"...]
        shots_num : Number of shots for averaging the readout
    """
    results = [RESULT_VAL] * RB_NUMBER
    goal: float = np.mean(results)
    results_std = [RESULTS_STD] * RB_NUMBER
    seqs = single_length_RB(RB_number=RB_NUMBER, RB_length=RB_LENGTH, target=TARGET)
    shots_nums = [SHOTS] * RB_NUMBER
    return goal, results, results_std, seqs, shots_nums


@pytest.mark.integration
def test_calibration_cmaes() -> None:
    """Create a C2 style Optimizer object and run calibration
    with a mock_ORBIT function. Check if the goal in the optimizer
    correctly reflects the constant goal returned by the mock_ORBIT.
    """
    with open(OPT_CONFIG_FILE_NAME, "r") as cfg_file:
        cfg = hjson.load(cfg_file)

    pmap = ParameterMap()
    pmap.read_config(cfg.pop("instructions"))
    pmap.set_opt_map(
        [[tuple(par) for par in pset] for pset in cfg.pop("gateset_opt_map")]
    )

    exp = Experiment(pmap=pmap)

    algo_options = cfg.pop("options")
    run_name = cfg.pop("run_name")

    opt = C2(
        dir_path=LOGDIR,
        run_name=run_name,
        eval_func=mock_ORBIT,
        pmap=pmap,
        exp_right=exp,
        algorithm=algorithms.cmaes,
        options=algo_options,
    )

    opt.run()
    assert opt.current_best_goal == RESULT_VAL


@pytest.mark.integration
def test_run_c2_config() -> None:
    """Run a C2 style Optimization task from a config file.
    This checks for the integrity of the c2 style config file
    The eval_func is set at run time to mock_ORBIT
    """
    with open(OPT_CONFIG_FILE_NAME, "r") as cfg_file:
        cfg = hjson.load(cfg_file)
        cfg.pop("eval_func")
        cfg["eval_func"] = mock_ORBIT
    run_cfg(cfg, "test/c2.cfg", debug=False)
