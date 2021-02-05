"""Various utility functions to speed up tensorflow coding."""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
from c3.utils import qt_utils


def tf_setup():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def tf_log_level_info():
    """Display the information about different log levels in tensorflow."""
    info = (
        "Log levels of tensorflow:\n"
        "\t0 = all messages are logged (default behavior)\n"
        "\t1 = INFO messages are not printed\n"
        "\t2 = INFO and WARNING messages are not printed\n"
        "\t3 = INFO, WARNING, and ERROR messages are not printed\n"
    )
    print(info)


def get_tf_log_level():
    """Display the current tensorflow log level of the system."""
    log_lvl = "0"

    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        log_lvl = os.environ["TF_CPP_MIN_LOG_LEVEL"]

    return log_lvl


def set_tf_log_level(lvl):
    """
    Set tensorflows system log level.

    REMARK: it seems like the 'TF_CPP_MIN_LOG_LEVEL' variable expects a string.
            the input of this function seems to work with both string and/or
            integer, as casting string to string does nothing. feels hacked?
            but I guess it's just python...
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(lvl)


def tf_list_avail_devices():
    """
    List available devices.

    Function for displaying all available devices for tf_setuptensorflow
    operations on the local machine.

    TODO:   Refine output of this function. But without further knowledge
            about what information is needed, best practise is to output all
            information available.
    """
    local_dev = device_lib.list_local_devices()
    print(local_dev)


def tf_limit_gpu_memory(memory_limit):
    """
    Set a limit for the GPU memory.

    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )
                ],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def tf_measure_operator(M, rho):
    """
    Expectation value of a quantum operator by tracing with a density matrix.

    Parameters
    ----------
    M : tf.tensor
        A quantum operator.
    rho : tf.tensor
        A density matrix.

    Returns
    -------
    tf.tensor
        Expectation value.

    """
    return tf.linalg.trace(tf.matmul(M, rho))


@tf.function
def tf_dU_of_t(h0, hks, cflds_t, dt):
    """
    Compute H(t) = H_0 + sum_k c_k H_k and matrix exponential exp(i H(t) dt).

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
    cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
    hks = tf.cast(hks, dtype=tf.complex128)
    cflds = tf.expand_dims(tf.expand_dims(cflds_t, 2), 3)
    hks = tf.expand_dims(hks, 1)
    h0 = tf.expand_dims(h0, 0)
    prod = cflds * hks
    h = h0 + tf.reduce_sum(prod, axis=0)
    dh = -1.0j * h * dt
    dU = tf.linalg.expm(dh)
    return dU


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


# def tf_propagation(h0, hks, cflds, dt):
#    """
#    Time evolution of a system controlled by time-dependent fields.
#
#    Parameters
#    ----------
#    h0 : tf.tensor
#        Drift Hamiltonian.
#    hks : list of tf.tensor
#        List of control Hamiltonians.
#    cflds : list
#        List of control fields, one per control Hamiltonian.
#    dt : float
#        Length of one time slice.
#
#    Returns
#    -------
#    type
#        Description of returned object.
#
#    """
#    def tf_time_slice(cf_t):
#        return tf_dU_of_t(h0, hks, cf_t, dt)
#
#    cflds = tf.cast(tf.transpose(tf.stack(cflds)), tf.complex128)
#    return tf.map_fn(tf_time_slice, cflds)

# EXPERIMENTAL BATCH PROPAGATION BELOW

# def tf_propagation(h0, hks, cflds, dt):
#     """
#     Calculate the time evolution of a system controlled by time-dependent
#     fields.
#
#     Parameters
#     ----------
#     h0 : tf.tensor
#         Drift Hamiltonian.
#     hks : list of tf.tensor
#         List of control Hamiltonians.
#     cflds : list
#         List of control fields, one per control Hamiltonian.
#     dt : float
#         Length of one time slice.
#
#     Returns
#     -------
#     type
#         Description of returned object.
#
#     """
#     dUs = []
#     batch_size = 4
#     for ii in range(cflds[0].shape[0]//batch_size):
#         dUs.extend(
#             tf_propagation_batch(h0, hks, cflds, dt, ii)
#         )
#     return dUs
#
#
#
# def tf_propagation_batch(h0, hks, cflds, dt, left):
#     """
#     """
#     dUs = []
#     for ii in range(left, left+4):
#         cf_t = []
#         for fields in cflds:
#             cf_t.append(tf.cast(fields[ii], tf.complex128))
#         dUs.append(tf_dU_of_t(h0, hks, cf_t, dt))
#     return dUs


def tf_propagation_lind(h0, hks, col_ops, cflds, dt, history=False):
    """
    Calculate the time evolution of an open system controlled by time-dependent
    fields.

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    col_ops : list of tf.tensor
        List of collapse operators.
    cflds : list
        List of control fields, one per control Hamiltonian.
    dt : float
        Length of one time slice.

    Returns
    -------
    list
        List of incremental propagators dU.

    """
    with tf.name_scope("Propagation"):
        dUs = []
        for ii in range(len(cflds[0])):
            cf_t = []
            for fields in cflds:
                cf_t.append(tf.cast(fields[ii], tf.complex128))
            dUs.append(tf_dU_of_t_lind(h0, hks, col_ops, cf_t, dt))
        return dUs


# MATRIX MULTIPLICATION FUNCTIONS


def evaluate_sequences(U_dict: dict, sequences: list):
    """
    Compute the total propagator of a sequence of gates.

    Parameters
    ----------
    U_dict : dict
        Dictionary of unitary representation of gates.

    sequences : list
        List of keys from U_dict specifying a gate sequence.
        The sequence is multiplied from the left, i.e.
            sequence = [U0, U1, U2, ...]
        is applied as
            ... U2 * U1 * U0

    Returns
    -------
    tf.tensor
        Propagator of the sequence.

    """
    gates = U_dict
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


# def tf_matmul_left(dUs):
#     """
#     Multiplies a list of matrices from the left.
#
#     """
#     U = dUs[0]
#     for ii in range(1, len(dUs)):
#         U = tf.matmul(dUs[ii], U, name="timestep_" + str(ii))
#     return U


@tf.function
def tf_matmul_left(dUs: tf.Tensor):
    """
    Parameters:
        dUs: tf.Tensor
            Tensorlist of shape (N, n,m)
            with number N matrices of size nxm
    Multiplies a list of matrices from the left.

    """
    return tf.foldr(lambda a, x: tf.matmul(a, x), dUs)


# def tf_matmul_right(dUs):
#     """
#     Multiplies a list of matrices from the right.
#
#     """
#     U = dUs[0]
#     for ii in range(1, len(dUs)):
#         U = tf.matmul(U, dUs[ii], name="timestep_" + str(ii))
#     return U


@tf.function
def tf_matmul_right(dUs):
    """
    Parameters:
        dUs: tf.Tensor
            Tensorlist of shape (N, n,m)
            with number N matrices of size nxm
    Multiplies a list of matrices from the right.

    """
    return tf.foldl(lambda a, x: tf.matmul(a, x), dUs)


def tf_matmul_n(tensor_list):
    """
    Multiply a list of tensors as binary tree.

    EXPERIMENTAL
    """
    # TODO does it multiply from the left?
    ln = len(tensor_list)
    if ln == 1:
        return tensor_list[0]
    else:
        left_half = tensor_list[0 : int(ln / 2)]
        right_half = tensor_list[int(ln / 2) : ln]
        return tf.matmul(tf_matmul_n(left_half), tf_matmul_n(right_half))


# MATH FUNCTIONS
def tf_log10(x):
    """Tensorflow had no logarithm with base 10. This is ours."""
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_abs_squared(x):
    """Rewritten so that is has a gradient."""
    return tf.reshape(tf.cast(tf.math.conj(x) * x, dtype=tf.float64), shape=[1])


def tf_abs(x):
    """Rewritten so that is has a gradient."""
    # TODO: See if custom abs and abs_squared are needed and compare performance.
    return tf.sqrt(tf_abs_squared(x))


def tf_ave(x: list):
    """Take average of a list of values in tensorflow."""
    return tf.add_n(x) / len(x)


def tf_diff(l):  # noqa
    """
    Running difference of the input list l. Equivalent to np.diff, except it
    returns the same shape by adding a 0 in the last entry.
    """
    dim = l.shape[0] - 1
    diagonal = tf.constant([-1] * dim + [0], dtype=l.dtype)
    offdiagonal = tf.constant([1] * dim, dtype=l.dtype)
    proj = tf.linalg.diag(diagonal) + tf.linalg.diag(offdiagonal, k=1)
    return tf.linalg.matvec(proj, l)


# MATRIX FUNCTIONS


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
    r = tf.eye(int(A.shape[0]), dtype=A.dtype)
    A_powers = A
    r += A

    for ii in range(2, terms):
        A_powers = tf.matmul(A_powers, A) / tf.cast(ii, tf.complex128)
        ii += 1
        r += A_powers
    return r


def tf_expm_dynamic(A, acc=1e-4):
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


def Id_like(A):
    """Identity of the same size as A."""
    shape = tf.shape(A)
    dim = shape[0]
    return tf.eye(dim, dtype=tf.complex128)


def tf_kron(A, B):
    """Kronecker product of 2 matrices."""
    # TODO make kronecker product general to different dimensions
    dims = tf.shape(A) * tf.shape(B)
    tensordot = tf.tensordot(A, B, axes=0)
    reshaped = tf.reshape(tf.transpose(tensordot, perm=[0, 2, 1, 3]), dims)
    return reshaped


# SUPEROPER FUNCTIONS
# TODO migrate all superoper functions to using tf_kron


def tf_spre(A):
    """Superoperator on the left of matrix A."""
    Id = Id_like(A)
    dim = tf.shape(A)[0]
    tensordot = tf.tensordot(A, Id, axes=0)
    reshaped = tf.reshape(
        tf.transpose(tensordot, perm=[0, 2, 1, 3]), [dim ** 2, dim ** 2]
    )
    return reshaped


def tf_spost(A):
    """Superoperator on the right of matrix A."""
    Id = Id_like(A)
    dim = tf.shape(A)[0]
    tensordot = tf.tensordot(Id, tf.transpose(A), axes=0)
    reshaped = tf.reshape(
        tf.transpose(tensordot, perm=[0, 2, 1, 3]), [dim ** 2, dim ** 2]
    )
    return reshaped


def tf_super(A):
    """Superoperator from both sides of matrix A."""
    superA = tf.matmul(tf_spre(A), tf_spost(tf.linalg.adjoint(A)))
    return superA


def tf_choi_to_chi(U, dims=None):
    """
    Convert the choi representation of a process to chi representation.

    """
    if dims is None:
        dims = [tf.sqrt(tf.cast(U.shape[0], U.dtype))]
    B = tf.constant(qt_utils.pauli_basis([2] * len(dims)), dtype=tf.complex128)
    return tf.linalg.adjoint(B) @ U @ B


def super_to_choi(A):
    """
    Convert a super operator to choi representation.

    """
    sqrt_shape = int(np.sqrt(A.shape[0]))
    A_choi = tf.reshape(
        tf.transpose(tf.reshape(A, [sqrt_shape] * 4), perm=[3, 1, 2, 0]), A.shape
    )
    return A_choi


def tf_state_to_dm(psi_ket):
    """Make a state vector into a density matrix."""
    psi_ket = tf.reshape(psi_ket, [psi_ket.shape[0], 1])
    psi_bra = tf.transpose(psi_ket)
    return tf.matmul(psi_ket, psi_bra)


# TODO see which code to get dv is better (and kill the other)
def tf_dm_to_vec(dm):
    """Convert a density matrix into a density vector."""
    return tf.reshape(tf.transpose(dm), shape=[-1, 1])


def tf_vec_to_dm(vec):
    """Convert a density vector to a density matrix."""
    dim = tf.sqrt(tf.cast(vec.shape[0], tf.float32))
    return tf.transpose(tf.reshape(vec, [dim, dim]))


def tf_dmdm_fid(rho, sigma):
    """Trace fidelity between two density matrices."""
    # TODO needs fixing
    rhosqrt = tf.linalg.sqrtm(rho)
    return tf.linalg.trace(
        tf.linalg.sqrtm(tf.matmul(tf.matmul(rhosqrt, sigma), rhosqrt))
    )


def tf_dmket_fid(rho, psi):
    """Fidelity between a state vector and a density matrix."""
    return tf.sqrt(tf.matmul(tf.matmul(tf.linalg.adjoint(psi), rho), psi))


def tf_ketket_fid(psi1, psi2):
    """Overlap of two state vectors."""
    return tf_abs(tf.matmul(tf.linalg.adjoint(psi1), psi2))


def tf_unitary_overlap(A: tf.Tensor, B: tf.Tensor, lvls: tf.Tensor = None) -> tf.Tensor:
    """Unitary overlap between two matrices.

    Parameters
    ----------
    A : tf.Tensor
        Unitary A
    B : tf.Tensor
        Unitary B
    lvls : tf.Tensor, optional
        Levels, by default None

    Returns
    -------
    tf.Tensor
        Overlap between the two unitaries

    Raises
    ------
    TypeError
        For errors during cast
    ValueError
        For errors during matrix multiplicaton
    """
    try:
        if lvls is None:
            lvls = tf.cast(B.shape[0], B.dtype)
        overlap = (
            tf_abs(tf.linalg.trace(tf.matmul(A, tf.linalg.adjoint(B))) / lvls) ** 2
        )
    except TypeError:
        raise TypeError("Possible Inconsistent Dimensions while casting tensors")
    except ValueError:
        raise ValueError(
            "Possible Inconsistent Dimensions during Matrix Multiplication"
        )
    return overlap


def tf_superoper_unitary_overlap(A, B, lvls=None):
    # TODO: This is just wrong, probably.
    if lvls is None:
        lvls = tf.sqrt(tf.cast(B.shape[0], B.dtype))
    overlap = (
        tf_abs(tf.sqrt(tf.linalg.trace(tf.matmul(A, tf.linalg.adjoint(B)))) / lvls) ** 2
    )

    return overlap


def tf_average_fidelity(A, B, lvls=None):
    """A very useful but badly named fidelity measure."""
    if lvls is None:
        lvls = tf.cast(B.shape[0], B.dtype)
    Lambda = tf.matmul(
        tf.linalg.adjoint(tf_project_to_comp(A, lvls, to_super=False)), B
    )
    return tf_super_to_fid(tf_super(Lambda), lvls)


def tf_superoper_average_fidelity(A, B, lvls=None):
    """A very useful but badly named fidelity measure."""
    if lvls is None:
        lvls = tf.sqrt(tf.cast(B.shape[0], B.dtype))
    lambda_super = tf.matmul(tf.linalg.adjoint(tf_project_to_comp(A, lvls, True)), B)
    return tf_super_to_fid(lambda_super, lvls)


def tf_super_to_fid(err, lvls):
    """Return average fidelity of a process."""
    lambda_chi = tf_choi_to_chi(super_to_choi(err), dims=lvls)
    d = 2 ** len(lvls)
    # get only 00 element and measure fidelity
    return tf_abs((lambda_chi[0, 0] / d + 1) / (d + 1))


def tf_project_to_comp(A, dims, to_super=False):
    """Project an operator onto the computational subspace."""
    # TODO projection to computational subspace can be done more efficiently than this
    proj_list = []
    for dim in dims:
        p = np.zeros([dim, 2])
        p[0, 0] = 1
        p[1, 1] = 1
        if to_super:
            p = np.kron(p, p)
        proj_list.append(p)
    proj = proj_list.pop()
    while not proj_list == []:
        proj = np.kron(proj_list.pop(), proj)
    P = tf.constant(proj, dtype=A.dtype)
    return tf.matmul(tf.matmul(P, A, transpose_a=True), P)


# @tf.function
def tf_convolve(sig: tf.Tensor, resp: tf.Tensor):
    """
    Compute the convolution with a time response.

    Parameters
    ----------
    sig : tf.Tensor
        Signal which will be convoluted, shape: [N]
    resp : tf.Tensor
        Response function to be convoluted with signal, shape: [M]

    Returns
    -------
    tf.Tensor
        convoluted signal of shape [N]

    """
    sig = tf.cast(sig, dtype=tf.complex128)
    resp = tf.cast(resp, dtype=tf.complex128)

    sig_len = len(sig)
    resp_len = len(resp)

    signal_pad = tf.expand_dims(
        tf.concat([sig, tf.zeros(resp_len, dtype=tf.complex128)], axis=0), 0
    )
    resp_pad = tf.expand_dims(
        tf.concat([resp, tf.zeros(sig_len, dtype=tf.complex128)], axis=0), 0
    )
    sig_resp = tf.concat([signal_pad, resp_pad], axis=0)

    fft_sig_resp = tf.signal.fft(sig_resp)
    fft_conv = tf.math.reduce_prod(fft_sig_resp, axis=0)
    convolution = tf.signal.ifft(fft_conv)
    return convolution[:sig_len]
