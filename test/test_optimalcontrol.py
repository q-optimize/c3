import os
import tempfile

import numpy as np
import pytest

import c3.libraries.algorithms as algorithms
import c3.libraries.fidelities as fidelities
from c3.optimizers.optimalcontrol import OptimalControl, OptResultOOBError
from examples.single_qubit_experiment import create_experiment


@pytest.mark.integration
def test_raises_OOB_for_bad_optimizer() -> None:
    np.random.seed(0)

    exp = create_experiment()
    exp.set_opt_gates(['rx90p[0]'])

    opt_gates = ["rx90p[0]"]
    gateset_opt_map = [
        [
            ("rx90p[0]", "d1", "carrier", "framechange"),
        ],
        [
            ("rx90p[0]", "d1", "gauss", "amp"),
        ],
        [
            ("rx90p[0]", "d1", "gauss", "freq_offset"),
        ],
    ]

    exp.pmap.set_opt_map(gateset_opt_map)

    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")
    cmaes_options = {'spread': 1, 'maxiter': 5}

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=fidelities.unitary_infid_set,
        fid_subspace=["Q1"],
        pmap=exp.pmap,
        algorithm=algorithms.cmaes,
        options=cmaes_options,
        run_name="OOB_CMAES_opt"
    )

    exp.set_opt_gates(opt_gates)
    opt.set_exp(exp)

    try:
        opt.optimize_controls()
        assert False, 'Should have raised OptResultOOBError.'
    except OptResultOOBError:
        pass
