"""
integration testing module for C1 optimization through two-qubits example
"""

import pickle
import numpy as np
import pytest
import tensorflow as tf

from numpy.testing import assert_array_almost_equal as almost_equal
import c3.libraries.algorithms as algorithms
from c3.libraries.fidelities import state_transfer_from_states

# Libs and helpers
from c3.libraries.propagation import rk4_unitary

with open("test/two_qubit_data.pickle", "rb") as filename:
    test_data = pickle.load(filename)


@pytest.mark.tensorflow
@pytest.mark.unit
def test_signals(get_two_qubit_chip) -> None:
    """Compare generated control signals to stored examples."""
    exp = get_two_qubit_chip
    pmap = exp.pmap
    gen_signal = pmap.generator.generate_signals(pmap.instructions["rx90p[0]"])
    ts = gen_signal["d1"]["ts"]
    np.testing.assert_allclose(ts, test_data["ts"])
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["signal"]["d1"]["values"].numpy(),
    )


@pytest.mark.unit
def test_hamiltonians(get_two_qubit_chip) -> None:
    """Compare Hamilonians"""
    model = get_two_qubit_chip.pmap.model
    hdrift, hks = model.get_Hamiltonians()
    assert (hdrift.numpy() - test_data["hdrift"].numpy() < 1).any()
    for key in hks:
        almost_equal(hks[key], test_data["hks"][key])


@pytest.mark.tensorflow
@pytest.mark.integration
def test_propagation(get_two_qubit_chip) -> None:
    """Test that result of the propagation code does not change."""
    GATE_STR = "rx90p[0]"
    exp = get_two_qubit_chip
    pmap = exp.pmap
    instr = pmap.instructions[GATE_STR]
    steps = int((instr.t_end - instr.t_start) * exp.sim_res)
    result = exp.propagation(
        pmap.model,
        pmap.generator,
        pmap.instructions["rx90p[0]"],
        exp.folding_stack[steps],
    )
    propagator = result["U"]
    almost_equal(propagator, test_data["propagator"])


@pytest.mark.unit
def test_init_point(get_OC_optimizer) -> None:
    """Check that a previous best point can be loaded as an initial point."""
    opt = get_OC_optimizer
    opt.load_best("test/best_point_open_loop.c3log")


@pytest.mark.slow
@pytest.mark.tensorflow
@pytest.mark.optimizers
@pytest.mark.integration
def test_optim(get_OC_optimizer) -> None:
    """
    check if optimization result is below 1e-1
    """
    opt = get_OC_optimizer
    assert opt.evaluation == 0
    opt.optimize_controls()
    assert opt.current_best_goal < 0.1
    maxiterKey = "maxiters" if opt.algorithm == algorithms.tf_sgd else "maxiter"
    assert opt.evaluation == opt.options[maxiterKey] - 1


@pytest.mark.slow
@pytest.mark.tensorflow
@pytest.mark.optimizers
@pytest.mark.integration
def test_optim_ode_solver(get_OC_optimizer) -> None:
    """
    check if optimization result is below 1e-1
    """
    opt = get_OC_optimizer
    opt.fid_func = state_transfer_from_states
    opt.set_goal_function(ode_solver="rk4", ode_step_function="schrodinger")

    model = opt.exp.pmap.model
    psi_init = [[0] * model.tot_dim]
    psi_init[0][0] = 1 / np.sqrt(2)
    index = model.get_state_indeces([(1, 0)])[0]
    psi_init[0][index] = 1 / np.sqrt(2)
    target_state = tf.transpose(tf.constant(psi_init, tf.complex128))
    params = {"target": target_state}
    opt.fid_func_kwargs = {"params": params}

    assert opt.evaluation == 0
    opt.optimize_controls()

    if opt.algorithm == algorithms.single_eval:
        assert opt.current_best_goal < 0.5
    else:
        assert opt.current_best_goal < 0.3

    maxiterKey = "maxiters" if opt.algorithm == algorithms.tf_sgd else "maxiter"
    assert opt.evaluation == opt.options[maxiterKey] - 1


@pytest.mark.slow
@pytest.mark.tensorflow
@pytest.mark.optimizers
@pytest.mark.integration
def test_optim_ode_solver_final(get_OC_optimizer) -> None:
    """
    check if optimization result is below 1e-1
    """
    opt = get_OC_optimizer
    opt.fid_func = state_transfer_from_states
    opt.set_goal_function(
        ode_solver="rk4", ode_step_function="schrodinger", only_final_state=True
    )

    model = opt.exp.pmap.model
    psi_init = [[0] * model.tot_dim]
    psi_init[0][0] = 1 / np.sqrt(2)
    index = model.get_state_indeces([(1, 0)])[0]
    psi_init[0][index] = 1 / np.sqrt(2)
    target_state = tf.transpose(tf.constant(psi_init, tf.complex128))
    params = {"target": target_state}
    opt.fid_func_kwargs = {"params": params}

    assert opt.evaluation == 0
    opt.optimize_controls()
    if opt.algorithm == algorithms.single_eval:
        assert opt.current_best_goal < 0.5
    else:
        assert opt.current_best_goal < 0.2

    maxiterKey = "maxiters" if opt.algorithm == algorithms.tf_sgd else "maxiter"
    assert opt.evaluation == opt.options[maxiterKey] - 1


@pytest.mark.tensorflow
@pytest.mark.integration
def test_rk4_unitary(get_two_qubit_chip) -> None:
    """Testing that RK4 exists and runs."""
    exp = get_two_qubit_chip
    pmap = exp.pmap
    exp.set_prop_method(rk4_unitary)
    exp.propagation(pmap.model, pmap.generator, pmap.instructions["rx90p[0]"])


@pytest.mark.tensorflow
@pytest.mark.integration
def test_ode_solver(get_two_qubit_chip) -> None:
    """Testing that ODE solver exists and runs for solvers rk4, rk5, tsit5."""
    exp = get_two_qubit_chip
    exp.set_opt_gates(["rx90p[0]"])
    exp.compute_states(solver="rk4")
    exp.compute_states(solver="rk5")
    exp.compute_states(solver="tsit5")
    exp.compute_states(solver="rk4", step_function="von_neumann")
    exp.compute_states(solver="rk5", step_function="von_neumann")
    exp.compute_states(solver="tsit5", step_function="von_neumann")


@pytest.mark.tensorflow
@pytest.mark.integration
def test_compute_final_state(get_two_qubit_chip) -> None:
    """Testing that compute_final_state exists and runs for solvers rk4, rk5, tsit5."""
    exp = get_two_qubit_chip
    exp.set_opt_gates(["rx90p[0]"])
    exp.compute_final_state(solver="rk4")
    exp.compute_final_state(solver="rk5")
    exp.compute_final_state(solver="tsit5")
    exp.compute_final_state(solver="rk4", step_function="von_neumann")
    exp.compute_final_state(solver="rk5", step_function="von_neumann")
    exp.compute_final_state(solver="tsit5", step_function="von_neumann")


@pytest.mark.slow
@pytest.mark.tensorflow
@pytest.mark.integration
def test_lindblad_propagation(get_two_qubit_chip) -> None:
    """Test that result of the propagation code does not change."""
    GATE_STR = "rx90p[0]"
    exp = get_two_qubit_chip
    exp.propagate_batch_size = 360
    pmap = exp.pmap
    model = pmap.model
    model.set_lindbladian(True)
    instr = pmap.instructions[GATE_STR]
    steps = int((instr.t_end - instr.t_start) * exp.sim_res)
    result = exp.propagation(
        model,
        pmap.generator,
        pmap.instructions["rx90p[0]"],
        exp.folding_stack[steps],
    )
    propagator = result["U"]
    almost_equal(propagator, test_data["lindblad_propagator"])


@pytest.mark.tensorflow
@pytest.mark.integration
def test_ode_solver_lindblad(get_two_qubit_chip) -> None:
    """Testing that ODE solver for lindbladian exists and runs for solvers."""
    exp = get_two_qubit_chip
    model = exp.pmap.model
    model.set_lindbladian(True)
    exp.set_opt_gates(["rx90p[0]"])
    exp.compute_states(solver="rk4")
    exp.compute_final_state(solver="rk4")
