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
    tf_matmul_n,
    tf_spre,
    tf_spost,
    Id_like,
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
    if not np.all(tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])):  # type: ignore
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
def pwc(model: Model, gen: Generator, instr: Instruction, folding_stack: list) -> Dict:
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
    # Why do I get 0.0 if I rint gen.resolution here?! FR

    dynamics_generators = model.get_dynamics_generators(signal)

    dUs = tf.linalg.expm(dynamics_generators)

    # U = tf_matmul_left(tf.cast(dUs, tf.complex128))
    U = tf_matmul_n(dUs, folding_stack)

    if model.max_excitations:
        U = model.blowup_excitations(tf_matmul_left(tf.cast(dUs, tf.complex128)))
        dUs = tf.vectorized_map(model.blowup_excitations, dUs)

    return {"U": U, "dUs": dUs}


@unitary_deco
def pwc_sequential(model: Model, gen: Generator, instr: Instruction) -> Dict:
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
    # get number of time steps in the signal
    n = tf.constant(len(list(list(signal.values())[0].values())[0]), dtype=tf.int32)

    if model.lindbladian:
        U = Id_like(model.get_Liouvillian(None))
    else:
        U = Id_like(model.get_Hamiltonian())

    for i in range(n):
        mini_signal: Dict[str, Dict] = {}
        for key in signal.keys():
            mini_signal[key] = {}
            mini_signal[key]["values"] = tf.expand_dims(signal[key]["values"][i], 0)
            # the ts are only used to compute dt and therefore this works
            mini_signal[key]["ts"] = signal[key]["ts"][0:2]
        dynamics_generators = model.get_dynamics_generators(mini_signal)

        dU = tf.linalg.expm(dynamics_generators)
        # i made shure that this order produces the same result as the original pwc function
        U = dU[0] @ U

    if model.max_excitations:
        U = model.blowup_excitations(U)

    return {"U": U}


@unitary_deco
def pwc_sequential_parallel(
    model: Model,
    gen: Generator,
    instr: Instruction,
    parallel: int = 16,
) -> Dict:
    """
    Solve the equation of motion (Lindblad or Schrรถdinger) for a given control
    signal and Hamiltonians.
    This function will be retraced if different values of parallel are input
    since parallel is input as a non tensorflow datatype

    Parameters
    ----------
    signal: dict
        Waveform of the control signal per drive line.
    gate: str
        Identifier for one of the gates.
    parallel: int
        number of prarallelly executing matrix multiplications

    Returns
    -------
    unitary
        Matrix representation of the gate.
    """
    # In this function, there is a deliberate and clear distinction between tensorflow
    # and non tensorflow datatypes to guide which parts are hardwired during tracing
    # and which are not. This was necessary to allow for the tf.function decorator.
    signal = gen.generate_signals(instr)
    # get number of time steps in the signal. Since this should always be the same,
    # it does not interfere with tf.function tracing despite its numpy data type
    n_np = len(list(list(signal.values())[0].values())[0])  # non tensorflow datatype

    # batch_size is the number of operations happening sequentially, parallel is the number
    # of operations happening in parallel.
    # Their product is n or slightly bigger than n. n is the total number of operations.
    batch_size_np = np.ceil(n_np / parallel)  # non tensorflow datatype
    batch_size = tf.constant(batch_size_np, dtype=tf.int32)  # tensorflow datatype

    # i tried to set the Us to None at the beginning and have an if else condition
    # that handles the first call, but tensorflow complained

    # edge case at the end must be handled outside the loop so that tensorflow can propperly
    # trace the loop. I use this to simultaniously initialize the Us
    mismatch = int(
        n_np - np.floor(n_np / parallel) * parallel
    )  # a modulo might also work
    mini_init_signal: Dict[str, Dict] = {}
    for key in signal.keys():
        mini_init_signal[key] = {}
        # the signals pulled here are not in sequence, but that should not matter
        # the multiplication of the relevant propagators is still ordered correctly
        mini_init_signal[key]["values"] = signal[key]["values"][
            batch_size - 1 :: batch_size
        ]
        # this does nothing but reashure tensorflow of the shape of the tensor
        mini_init_signal[key]["values"] = tf.reshape(
            mini_init_signal[key]["values"], [mismatch]
        )
        # the ts are only used to compute dt and therefore this works
        mini_init_signal[key]["ts"] = signal[key]["ts"][0:2]
    dynamics_generators = model.get_dynamics_generators(mini_init_signal)
    Us = tf.linalg.expm(dynamics_generators)
    # possibly correct shape
    if Us.shape[1] is not parallel:
        Us = tf.concat(
            [
                Us,
                tf.eye(
                    Us.shape[1],
                    batch_shape=[parallel - Us.shape[0]],
                    dtype=tf.complex128,
                ),
            ],
            axis=0,
        )

    # since we had to start from the back to handle the final batch with possibly different length
    # we need to continue backwards and reverse the order (see batch_size - i - 2)
    # this is necessary because "reversed" cant be traced
    for i in range(batch_size - 1):
        mini_signal: Dict[str, Dict] = {}
        for key in signal.keys():
            mini_signal[key] = {}
            # the signals pulled here are not in sequence, but that should not matter
            # the multiplication of the relevant propagators is still ordered correctly
            mini_signal[key]["values"] = signal[key]["values"][
                (batch_size - i - 2) :: batch_size
            ]
            # this does nothing but reashure tensorflow of the shape of the tensor
            mini_signal[key]["values"] = tf.reshape(
                mini_signal[key]["values"], [parallel]
            )
            # the ts are only used to compute dt and therefore this works
            mini_signal[key]["ts"] = signal[key]["ts"][0:2]
        dynamics_generators = model.get_dynamics_generators(mini_signal)

        dUs = tf.linalg.expm(dynamics_generators)
        # i made shure that this order produces the same result as the original pwc function
        # though it is reversed just like the for loop is reversed
        Us = Us @ dUs

    # The Us are partially accumulated propagators, multiplying them together
    # yields the final propagator. They serve a similar function as the dUs
    # but there are typically fewer of them
    # here the multiplication order is flipped compared to the Us above.
    # while each individual propagator in Us was accumulatd in reverse,
    # the thereby resulting Us are still orderd advancing in time
    U = tf_matmul_left(tf.cast(Us, tf.complex128))

    if model.max_excitations:
        U = model.blowup_excitations(tf_matmul_left(tf.cast(Us, tf.complex128)))
        Us = tf.vectorized_map(model.blowup_excitations, Us)

    return {"U": U, "dUs": Us}


####################
# HELPER FUNCTIONS #
####################


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
    dcomm = -comm * dt**2 / 2.0
    dUs = dU0 @ tf.linalg.expm(dh) @ (tf.identity(dU0) - dcomm)
    return dUs


def tf_batch_propagate(dyn_gens, batch_size):
    """
    Propagate signal in batches
    Parameters
    ----------
    dyn_gens: tf.tensor
        i) -1j * Hamiltonian(t) * dt
        or
        ii) -1j * Liouville superoperator * dt
    batch_size: int
        Number of elements in one batch

    Returns
    -------
    List of partial propagators
    i) as operators
    ii) as superoperators
    """

    batches = int(tf.math.ceil(dyn_gens.shape[0] / batch_size))
    batch_array = tf.TensorArray(
        dyn_gens.dtype, size=batches, dynamic_size=False, infer_shape=False
    )
    for i in range(batches):
        batch_array = batch_array.write(
            i, dyn_gens[i * batch_size : i * batch_size + batch_size]
        )

    dUs_array = tf.TensorArray(tf.complex128, size=batches, infer_shape=False)
    for i in range(batches):
        x = batch_array.read(i)
        result = tf.linalg.expm(tf.cast(x, dtype=tf.complex128))
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
