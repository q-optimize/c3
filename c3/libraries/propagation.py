"A library for propagators and closely related functions"
import numpy as np
import tensorflow as tf
from typing import Dict
from c3.model import Model
from c3.generator.generator import Generator
from c3.signal.gates import Instruction
from c3.utils.tf_utils import (
    tf_kron,
    tf_matmul_left,
    tf_spre,
    tf_spost,
)

unitary_provider = dict()
state_provider = dict()


def step_vonNeumann_psi(psi, h, dt):
    return -1j * dt * tf.linalg.matvec(h, psi)


def unitary_deco(func):
    """
    Decorator for making registry of functions
    """
    unitary_provider[str(func.__name__)] = func
    return func


def state_deco(func):
    """
    Decorator for making registry of functions
    """
    state_provider[str(func.__name__)] = func
    return func


@unitary_deco
def gen_dus_rk4(h, dt, dim=None):
    dUs = []
    dU = []
    if dim is None:
        tot_dim = tf.shape(h)
        dim = tot_dim[1]

    for jj in range(0, len(h) - 2, 2):
        dU = gen_du_rk4(h[jj : jj + 3], dt, dim)
        dUs.append(dU)
    return dUs


def gen_du_rk4(h, dt, dim):
    temp = []
    for ii in range(dim):
        psi = tf.one_hot(ii, dim, dtype=tf.complex128)
        psi = rk4_step(h, psi, dt)
        temp.append(psi)
    dU = tf.stack(temp)
    return dU


def rk4_step(h, psi, dt):
    k1 = step_vonNeumann_psi(psi, h[0], dt)
    k2 = step_vonNeumann_psi(psi + k1 / 2.0, h[1], dt)
    k3 = step_vonNeumann_psi(psi + k2 / 2.0, h[1], dt)
    k4 = step_vonNeumann_psi(psi + k3, h[2], dt)
    psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return psi


def get_hs_of_t_ts(
    model: Model, gen: Generator, instr: Instruction, prop_res=1
) -> Dict:
    if model.controllability:
        hs_of_ts = _get_hs_of_t_ts_controlled(model, gen, instr, prop_res)
    else:
        hs_of_ts = _get_hs_of_t_ts(model, gen, instr, prop_res)
    return hs_of_ts


def _get_hs_of_t_ts_controlled(
    model: Model, gen: Generator, instr: Instruction, prop_res=1
) -> Dict:
    """
    Return a Dict containing:

    - a list of

      H(t) = H_0 + sum_k c_k H_k.

    - time slices ts

    - timestep dt

    Parameters
    ----------
    prop_res : tf.float
        resolution required by the propagation method
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    ts : float
        Length of one time slice.
    """
    Hs = []
    ts = []
    gen.resolution = prop_res * gen.resolution
    signal = gen.generate_signals(instr)
    h0, hctrls = model.get_Hamiltonians()
    signals = []
    hks = []
    for key in signal:
        signals.append(signal[key]["values"])
        ts = signal[key]["ts"]
        hks.append(hctrls[key])
    cflds = tf.cast(signals, tf.complex128)
    hks = tf.cast(hks, tf.complex128)
    for ii in range(cflds[0].shape[0]):
        cf_t = []
        for fields in cflds:
            cf_t.append(tf.cast(fields[ii], tf.complex128))
        Hs.append(sum_h0_hks(h0, hks, cf_t))

    dt = tf.constant(ts[1 * prop_res].numpy() - ts[0].numpy(), dtype=tf.complex128)
    return {"Hs": Hs, "ts": ts[::prop_res], "dt": dt}


def _get_hs_of_t_ts(
    model: Model, gen: Generator, instr: Instruction, prop_res=1
) -> Dict:
    """
    Return a Dict containing:

    - a list of

      H(t) = H_0 + sum_k c_k H_k.

    - time slices ts

    - timestep dt

    Parameters
    ----------
    prop_res : tf.float
        resolution required by the propagation method
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    ts : float
        Length of one time slice.
    """
    Hs = []
    ts = []
    gen.resolution = prop_res * gen.resolution
    signal = gen.generate_signals(instr)
    Hs = model.get_Hamiltonian(signal)
    ts_list = [sig["ts"][1:] for sig in signal.values()]
    ts = tf.constant(tf.math.reduce_mean(ts_list, axis=0))
    if not np.all(tf.math.reduce_variance(ts_list, axis=0) < 1e-5 * (ts[1] - ts[0])):
        raise Exception("C3Error:Something with the times happend.")
    if not np.all(tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])):
        raise Exception("C3Error:Something with the times happend.")

    dt = tf.constant(ts[1 * prop_res].numpy() - ts[0].numpy(), dtype=tf.complex128)
    return {"Hs": Hs, "ts": ts[::prop_res], "dt": dt}


def sum_h0_hks(h0, hks, cf_t):
    """
    Compute and Return

     H(t) = H_0 + sum_k c_k H_k.
    """
    h_of_t = h0
    ii = 0
    while ii < len(hks):
        h_of_t += cf_t[ii] * hks[ii]
        ii += 1
    return h_of_t


@unitary_deco
def rk4(model: Model, gen: Generator, instr: Instruction, init_state=None) -> Dict:
    prop_res = 2
    dim = model.tot_dim
    Hs = []
    ts = []
    dUs = []
    dict_vals = get_hs_of_t_ts(model, gen, instr, prop_res)
    Hs = dict_vals["Hs"]
    ts = dict_vals["ts"]
    dt = dict_vals["dt"]

    dUs = gen_dus_rk4(Hs, dt, dim)

    U = gen_u_rk4(Hs, dt, dim)

    if model.max_excitations:
        U = model.blowup_excitations(U)
        dUs = tf.vectorized_map(model.blowup_excitations, dUs)

    return {"U": U, "dUs": dUs, "ts": ts}


def gen_u_rk4(h, dt, dim):
    U = []
    for ii in range(dim):
        psi = tf.one_hot(ii, dim, dtype=tf.complex128)

        for jj in range(0, len(h) - 2, 2):
            psi = rk4_step(h[jj : jj + 3], psi, dt)
        U.append(psi)
    U = tf.stack(U)
    return tf.transpose(U)


@unitary_deco
def pwc(model: Model, gen: Generator, instr: Instruction) -> Dict:
    """
    Solve the equation of motion (Lindblad or Schrรถdinger) for a given control
    signal and Hamiltonians.

    Parameters
    ----------
    signal: dict
        Waveform of the control signal per drive line.
    gate: str
        Identifier for one of the gates.

    Returns
    -------
    unitary
        Matrix representation of the gate.
    """
    signal = gen.generate_signals(instr)
    # Why do I get 0.0 if I print gen.resolution here?! FR
    ts = []
    if model.controllability:
        h0, hctrls = model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            ts = signal[key]["ts"]
            hks.append(hctrls[key])
        signals = tf.cast(signals, tf.complex128)
        hks = tf.cast(hks, tf.complex128)
    else:
        h0 = model.get_Hamiltonian(signal)
        ts_list = [sig["ts"][1:] for sig in signal.values()]
        ts = tf.constant(tf.math.reduce_mean(ts_list, axis=0))
        if not np.all(
            tf.math.reduce_variance(ts_list, axis=0) < 1e-5 * (ts[1] - ts[0])
        ):
            raise Exception("C3Error:Something with the times happend.")
        if not np.all(
            tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])
        ):
            raise Exception("C3Error:Something with the times happend.")

    dt = tf.constant(ts[1].numpy() - ts[0].numpy(), dtype=tf.complex128)

    batch_size = tf.constant(len(h0), tf.int32)

    dUs = tf_batch_propagate(h0, hks, signals, dt, batch_size=batch_size)

    U = tf_matmul_left(tf.cast(dUs, tf.complex128))

    if model.max_excitations:
        U = model.blowup_excitations(tf_matmul_left(tf.cast(dUs, tf.complex128)))
        dUs = tf.vectorized_map(model.blowup_excitations, dUs)

    return {"U": U, "dUs": dUs, "ts": ts}


####################
# HELPER FUNCTIONS #
####################


@tf.function
def tf_dU_of_t(h0, hks, cflds_t, dt):
    """
    Compute H(t) = H_0 + sum_k c_k H_k and matrix exponential exp(-i H(t) dt).

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    dt : float
        Length of one time slice.

    Returns
    -------
    tf.tensor
        dU = exp(-i H(t) dt)

    """
    h = h0
    ii = 0
    while ii < len(hks):
        h += cflds_t[ii] * hks[ii]
        ii += 1
    # terms = int(1e12 * dt) + 2
    # dU = tf_expm(-1j * h * dt, terms)
    # TODO Make an option for the exponentation method
    dU = tf.linalg.expm(-1j * h * dt)
    return dU


# @tf.function
def tf_dU_of_t_lind(h0, hks, col_ops, cflds_t, dt):
    """
    Compute the Lindbladian and it's matrix exponential exp(L(t) dt).

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    col_ops : list of tf.tensor
        List of collapse operators.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    dt : float
        Length of one time slice.

    Returns
    -------
    tf.tensor
        dU = exp(L(t) dt)

    """
    h = h0
    for ii in range(len(hks)):
        h += cflds_t[ii] * hks[ii]
    lind_op = -1j * (tf_spre(h) - tf_spost(h))
    for col_op in col_ops:
        super_clp = tf.matmul(tf_spre(col_op), tf_spost(tf.linalg.adjoint(col_op)))
        anticomm_L_clp = 0.5 * tf.matmul(
            tf_spre(tf.linalg.adjoint(col_op)), tf_spre(col_op)
        )
        anticomm_R_clp = 0.5 * tf.matmul(
            tf_spost(col_op), tf_spost(tf.linalg.adjoint(col_op))
        )
        lind_op = lind_op + super_clp - anticomm_L_clp - anticomm_R_clp
    # terms = int(1e12 * dt) # Eyeball number of terms in expm
    #     print('terms in exponential: ', terms)
    # dU = tf_expm(lind_op * dt, terms)
    # Built-in tensorflow exponential below
    dU = tf.linalg.expm(lind_op * dt)
    return dU


@tf.function
def tf_propagation_vectorized(h0, hks, cflds_t, dt):
    dt = tf.cast(dt, dtype=tf.complex128)
    if hks is not None and cflds_t is not None:
        cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
        hks = tf.cast(hks, dtype=tf.complex128)
        cflds = tf.expand_dims(tf.expand_dims(cflds_t, 2), 3)
        hks = tf.expand_dims(hks, 1)
        if len(h0.shape) < 3:
            h0 = tf.expand_dims(h0, 0)
        prod = cflds * hks
        h = h0 + tf.reduce_sum(prod, axis=0)
    else:
        h = tf.cast(h0, tf.complex128)
    dh = -1.0j * h * dt
    return tf.linalg.expm(dh)


def pwc_trott_drift(h0, hks, cflds_t, dt):
    dt = tf.cast(dt, dtype=tf.complex128)
    cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
    hks = tf.cast(hks, dtype=tf.complex128)
    e, v = tf.linalg.eigh(h0)
    ort = tf.cast(v, dtype=tf.complex128)
    dE = tf.math.exp(-1.0j * tf.math.real(e) * dt)
    dU0 = ort @ tf.linalg.diag(dE) @ ort.T
    prod = cflds_t * hks
    ht = tf.reduce_sum(prod, axis=0)
    comm = h0 @ ht - ht @ h0
    dh = -1.0j * ht * dt
    dcomm = -comm * dt ** 2 / 2.0
    dUs = dU0 @ tf.linalg.expm(dh) @ (tf.identity(dU0) - dcomm)
    return dUs


def tf_batch_propagate(hamiltonian, hks, signals, dt, batch_size):
    """
    Propagate signal in batches
    Parameters
    ----------
    hamiltonian: tf.tensor
        Drift Hamiltonian
    hks: Union[tf.tensor, List[tf.tensor]]
        List of control hamiltonians
    signals: Union[tf.tensor, List[tf.tensor]]
        List of control signals, one per control hamiltonian
    dt: float
        Length of one time slice
    batch_size: int
        Number of elements in one batch

    Returns
    -------

    """
    if signals is not None:
        batches = int(tf.math.ceil(signals.shape[0] / batch_size))
        batch_array = tf.TensorArray(
            signals.dtype, size=batches, dynamic_size=False, infer_shape=False
        )
        for i in range(batches):
            batch_array = batch_array.write(
                i, signals[i * batch_size : i * batch_size + batch_size]
            )
    else:
        batches = int(tf.math.ceil(hamiltonian.shape[0] / batch_size))
        batch_array = tf.TensorArray(
            hamiltonian.dtype, size=batches, dynamic_size=False, infer_shape=False
        )
        for i in range(batches):
            batch_array = batch_array.write(
                i, hamiltonian[i * batch_size : i * batch_size + batch_size]
            )

    dUs_array = tf.TensorArray(tf.complex128, size=batches, infer_shape=False)
    for i in range(batches):
        x = batch_array.read(i)
        if signals is not None:
            result = tf_propagation_vectorized(hamiltonian, hks, x, dt)
        else:
            result = tf_propagation_vectorized(x, None, None, dt)
        dUs_array = dUs_array.write(i, result)
    return dUs_array.concat()


@unitary_deco
def tf_propagation(h0, hks, cflds, dt):
    """
    Calculate the unitary time evolution of a system controlled by time-dependent
    fields.

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds : list
        List of control fields, one per control Hamiltonian.
    dt : float
        Length of one time slice.

    Returns
    -------
    list
        List of incremental propagators dU.

    """
    dUs = []

    for ii in range(cflds[0].shape[0]):
        cf_t = []
        for fields in cflds:
            cf_t.append(tf.cast(fields[ii], tf.complex128))
        dUs.append(tf_dU_of_t(h0, hks, cf_t, dt))
    return dUs


@tf.function
def tf_propagation_lind(h0, hks, col_ops, cflds_t, dt, history=False):
    col_ops = tf.cast(col_ops, dtype=tf.complex128)
    dt = tf.cast(dt, dtype=tf.complex128)
    if hks is not None and cflds_t is not None:
        cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
        hks = tf.cast(hks, dtype=tf.complex128)
        cflds = tf.expand_dims(tf.expand_dims(cflds_t, 2), 3)
        hks = tf.expand_dims(hks, 1)
        h0 = tf.expand_dims(h0, 0)
        prod = cflds * hks
        h = h0 + tf.reduce_sum(prod, axis=0)
    else:
        h = h0

    h_id = tf.eye(h.shape[-1], batch_shape=[h.shape[0]], dtype=tf.complex128)
    l_s = tf_kron(h, h_id)
    r_s = tf_kron(h_id, tf.linalg.matrix_transpose(h))
    lind_op = -1j * (l_s - r_s)

    col_ops_id = tf.eye(
        col_ops.shape[-1], batch_shape=[col_ops.shape[0]], dtype=tf.complex128
    )
    l_col_ops = tf_kron(col_ops, col_ops_id)
    r_col_ops = tf_kron(col_ops_id, tf.linalg.matrix_transpose(col_ops))

    super_clp = tf.matmul(l_col_ops, r_col_ops, adjoint_b=True)
    anticom_L_clp = 0.5 * tf.matmul(l_col_ops, l_col_ops, adjoint_a=True)
    anticom_R_clp = 0.5 * tf.matmul(r_col_ops, r_col_ops, adjoint_b=True)
    clp = tf.expand_dims(
        tf.reduce_sum(super_clp - anticom_L_clp - anticom_R_clp, axis=0), 0
    )
    lind_op += clp

    dU = tf.linalg.expm(lind_op * dt)
    return dU


def evaluate_sequences(propagators: Dict, sequences: list):
    """
    Compute the total propagator of a sequence of gates.

    Parameters
    ----------
    propagators : dict
        Dictionary of unitary representation of gates.

    sequences : list
        List of keys from propagators specifying a gate sequence.
        The sequence is multiplied from the left, i.e.
            sequence = [U0, U1, U2, ...]
        is applied as
            ... U2 * U1 * U0

    Returns
    -------
    tf.tensor
        Propagator of the sequence.

    """
    gates = propagators
    # get dims to deal with the case where a sequence is empty
    dim = list(gates.values())[0].shape[0]
    dtype = list(gates.values())[0].dtype
    # TODO deal with the case where you only evaluate one sequence
    U = []
    for sequence in sequences:
        if len(sequence) == 0:
            U.append(tf.linalg.eye(dim, dtype=dtype))
        else:
            Us = []
            for gate in sequence:
                Us.append(gates[gate])

            Us = tf.cast(Us, tf.complex128)
            U.append(tf_matmul_left(Us))
            # ### WARNING WARNING ^^ look there, it says left WARNING
    return U


def tf_expm(A, terms):
    """
    Matrix exponential by the series method.

    Parameters
    ----------
    A : tf.tensor
        Matrix to be exponentiated.
    terms : int
        Number of terms in the series.

    Returns
    -------
    tf.tensor
        expm(A)

    """
    r = tf.eye(int(A.shape[-1]), batch_shape=A.shape[:-2], dtype=A.dtype)
    A_powers = A
    r += A

    for ii in range(2, terms):
        A_powers = tf.matmul(A_powers, A) / tf.cast(ii, tf.complex128)
        ii += 1
        r += A_powers
    return r


def tf_expm_dynamic(A, acc=1e-5):
    """
    Matrix exponential by the series method with specified accuracy.

    Parameters
    ----------
    A : tf.tensor
        Matrix to be exponentiated.
    acc : float
        Accuracy. Stop when the maximum matrix entry reaches

    Returns
    -------
    tf.tensor
        expm(A)

    """
    r = tf.eye(int(A.shape[0]), dtype=A.dtype)
    A_powers = A
    r += A

    ii = tf.constant(2, dtype=tf.complex128)
    while tf.reduce_max(tf.abs(A_powers)) > acc:
        A_powers = tf.matmul(A_powers, A) / ii
        ii += 1
        r += A_powers
    return r
