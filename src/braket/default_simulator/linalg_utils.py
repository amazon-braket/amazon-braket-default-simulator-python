# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import itertools
from collections.abc import Sequence
from typing import Optional

import numba as nb
import numpy as np

_NEG_CONTROL_SLICE = slice(None, 1)
_CONTROL_SLICE = slice(1, None)
_NO_CONTROL_SLICE = slice(None, None)

BASIS_MAPPING = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

# Preallocate for up to 42 qubits
_SLICE_NONE_ARRAYS = {n: [slice(None)] * n for n in range(1, 43)}

_QUBIT_THRESHOLD = nb.int32(10)


class QuantumGateDispatcher:
    def __init__(self, n_qubits: int):
        """
        Makes a way to dispatch to different optimized functions based on qubit count.
        """
        self.n_qubits = n_qubits
        self.use_large = n_qubits > _QUBIT_THRESHOLD

        self.apply_single_qubit_gate = (
            _apply_single_qubit_gate_large if self.use_large else _apply_single_qubit_gate_small
        )

        self.apply_swap = _apply_swap_large if self.use_large else _apply_swap_small

        self.apply_controlled_phase_shift = (
            _apply_controlled_phase_shift_large
            if self.use_large
            else _apply_controlled_phase_shift_small
        )


def multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: Optional[tuple[int, ...]] = (),
    control_state: Optional[tuple[int, ...]] = (),
    out: Optional[np.ndarray] = None,
    dispatcher: Optional[QuantumGateDispatcher] = None,
) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits,
    controlling the operation as specified.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        controls (Optional[tuple[int]]): The qubits to control the operation on. Default ().
        control_state (Optional[tuple[int]]): A tuple of same length as `controls` with either
            a 0 or 1 in each index, corresponding to whether to control on the `|0⟩` or `|1⟩` state.
            Default (1,) * len(controls).
        out (Optional[np.ndarray]): Preallocated result array to reduce overhead of creating a new array each time.
        dispatcher(QuantumGateDispatcher): Dispatch to optimized functions based on qubit
            count.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    if dispatcher is None:
        dispatcher = QuantumGateDispatcher(state.size)

    if out is None:
        out = np.zeros_like(state, dtype=complex)

    if not controls:
        return _multiply_matrix(state, matrix, targets, out, dispatcher)

    control_state = control_state or (1,) * len(controls)

    ctrl_slices = [_NO_CONTROL_SLICE] * len(state.shape)
    for i, state_val in zip(controls, control_state):
        ctrl_slices[i] = _NEG_CONTROL_SLICE if state_val == 0 else _CONTROL_SLICE
    ctrl_tuple = tuple(ctrl_slices)

    np.copyto(out, state)

    controlled_slice = out[ctrl_tuple]
    _multiply_matrix(state[ctrl_tuple], matrix, targets, controlled_slice, dispatcher)

    return out


def _apply_single_qubit_gate_small(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
):
    """Applies single gates using array slicing."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = state.ndim

    slices_0 = _SLICE_NONE_ARRAYS[n_qubits].copy()
    slices_0[target] = 0
    slices_0_tuple = tuple(slices_0)

    slices_1 = _SLICE_NONE_ARRAYS[n_qubits].copy()
    slices_1[target] = 1
    slices_1_tuple = tuple(slices_1)

    out[slices_0_tuple] = a * state[slices_0_tuple] + b * state[slices_1_tuple]
    out[slices_1_tuple] = c * state[slices_0_tuple] + d * state[slices_1_tuple]

    return out


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_single_qubit_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
):
    """Applies single gates using bit masking."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = state.ndim
    total_size = state.size
    target_mask = 1 << (n_qubits - 1 - target)

    for i in nb.prange(total_size):
        idx0 = i & ~target_mask
        idx1 = i | target_mask

        if (i & target_mask) == 0:
            out.flat[i] = a * state.flat[idx0] + b * state.flat[idx1]
        else:
            out.flat[i] = c * state.flat[idx0] + d * state.flat[idx1]
    return out


def _apply_cnot(state: np.ndarray, control: int, target: int, out: np.ndarray) -> np.ndarray:
    """CNOT optimization path."""
    np.copyto(out, state)
    n_qubits = state.ndim

    slice_list = [slice(None)] * n_qubits

    slice_list[control] = 1
    slice_list[target] = 0
    slices_c1t0 = tuple(slice_list)

    slice_list[target] = 1
    slices_c1t1 = tuple(slice_list)

    out[slices_c1t0] = out[slices_c1t1]
    out[slices_c1t1] = state[slices_c1t0]

    return out


def _apply_swap_small(state: np.ndarray, qubit_0: int, qubit_1: int, out: np.ndarray) -> np.ndarray:
    """Swap gate implementation using numpy's swapaxes."""
    np.copyto(out, np.swapaxes(state, qubit_0, qubit_1))
    return out


@nb.njit(parallel=True, fastmath=True, cache=True)
def _apply_swap_large(
    state: np.ndarray, qubit_0: int, qubit_1: int, out: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Swap gate implementation using bit manipulation."""
    n_qubits = state.ndim
    total_size = 1 << n_qubits

    mask_0 = 1 << (n_qubits - 1 - qubit_0)
    mask_1 = 1 << (n_qubits - 1 - qubit_1)

    for i in nb.prange(total_size):
        j = i ^ mask_0 ^ mask_1

        bit_0 = (i & mask_0) != 0
        bit_1 = (i & mask_1) != 0
        use_partner = bit_0 ^ bit_1

        source_idx = use_partner * j + (1 - use_partner) * i
        out.flat[i] = state.flat[source_idx]

    return out


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_controlled_phase_shift_large(  # pragma: no cover
    state: np.ndarray, angle: float, controls, target: int, out: np.ndarray
) -> np.ndarray:
    """C Phase shift gate optimization path for larger vectors using bit masks.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        target (int): The qubit to apply the state on.
        out (np.ndarray): Output array for result.

    Returns:
        np.ndarray: The state array with the controlled phase shift gate applied.
    """
    phase_factor = np.exp(1j * angle)
    phase_factor_minus_one = phase_factor - 1.0
    n_qubits = state.ndim

    mask = 1 << (n_qubits - 1 - target)
    for c in controls:
        mask |= 1 << (n_qubits - 1 - c)

    for i in nb.prange(state.size):
        should_apply = (i & mask) == mask
        out.flat[i] = (1.0 + should_apply * phase_factor_minus_one) * state.flat[i]

    return out


def _apply_controlled_phase_shift_small(
    state: np.ndarray, angle: float, controls, target: int, out: np.ndarray
) -> np.ndarray:
    """C Phase shift gate optimization path for smaller vectors using numpy slicing."""
    phase_factor = np.exp(1j * angle)
    np.copyto(out, state)

    slices = _SLICE_NONE_ARRAYS[len(state.shape)].copy()
    for c in controls:
        slices[c] = 1
    slices[target] = 1

    out[tuple(slices)] *= phase_factor

    return out


def _apply_two_qubit_gate(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, int],
    out: np.ndarray,
    dispatcher: QuantumGateDispatcher,
) -> np.ndarray:
    """Two-qubit gates optimization path.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        out (np.ndarray): Output array for result.
        dispatcher(QuantumGateDispatcher): Dispatch to optimized functions based on qubit
            count.

    Returns:
        np.ndarray: The state after the matrix has been applied.

    """
    target0, target1 = targets
    n_qubits = state.ndim

    if matrix.ndim != 2 or matrix.shape != (4, 4):
        matrix = matrix.reshape(4, 4)

    # Moving away from np.allclose here to avoid slightly more expensive checks
    diag = np.diag(matrix)
    angle = np.angle(matrix[3, 3])

    if matrix[2, 3] == 1 and matrix[3, 2] == 1 and np.all(np.diag(matrix)[[0, 1]] == 1):
        return _apply_cnot(state, target0, target1, out)
    elif matrix[1, 2] == 1 and matrix[2, 1] == 1 and np.all(np.diag(matrix)[[0, 3]] == 1):
        return dispatcher.apply_swap(state, target0, target1, out)
    elif (
        abs(diag[0] - 1) < 1e-10
        and abs(diag[1] - 1) < 1e-10
        and abs(diag[2] - 1) < 1e-10
        and abs(diag[3] - np.exp(1j * angle)) < 1e-10
    ):
        return dispatcher.apply_controlled_phase_shift(state, angle, (target0,), target1, out)

    out.fill(0)

    slices = {}
    for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        slice_list = [slice(None)] * n_qubits
        slice_list[target0] = bits[0]
        slice_list[target1] = bits[1]
        slices[bits] = tuple(slice_list)

    rows, cols = np.nonzero(matrix)

    for k in range(len(rows)):
        i, j = rows[k], cols[k]
        coef = matrix[i, j]
        out_bits = BASIS_MAPPING[i]
        in_bits = BASIS_MAPPING[j]
        out[slices[out_bits]] += coef * state[slices[in_bits]]

    return out


def _multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    out: np.ndarray,
    dispatcher: QuantumGateDispatcher,
) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        out (np.ndarray): Output array for result.
        dispatcher(QuantumGateDispatcher): Dispatch to optimized functions based on qubit
            count.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    if len(targets) == 1:
        return dispatcher.apply_single_qubit_gate(state, matrix, targets[0], out)
    elif len(targets) == 2:
        return _apply_two_qubit_gate(state, matrix, targets, out, dispatcher)

    num_targets = len(targets)
    gate_matrix = np.reshape(matrix, [2] * num_targets * 2)
    axes = (np.arange(num_targets, 2 * num_targets), targets)

    product = np.tensordot(gate_matrix, state, axes=axes)
    unused_idxs = [idx for idx in range(state.ndim) if idx not in targets]

    np.copyto(out, np.transpose(product, np.argsort([*targets, *unused_idxs])))
    return out


def marginal_probability(
    probabilities: np.ndarray,
    targets: Sequence[int] = None,
) -> np.ndarray:
    """Return the marginal probability of the computational basis states.

    The marginal probability is obtained by summing the probabilities on
    the unused qubits. If no targets are specified, then the probability
    of all basis states is returned.

    Args:
        probabilities (np.ndarray): The probability distribution to marginalize.
        targets (list[int]): The qubits of the marginal distribution;
            if no targets are specified, then the probability of all basis states is returned.

    Returns:
        np.ndarray: The marginal probability distribution.
    """
    qubit_count = int(np.log2(len(probabilities)))

    if targets is None or np.array_equal(targets, range(qubit_count)):
        # All qubits targeted, no need to marginalize
        return probabilities

    targets = np.hstack(targets)

    # Find unused qubits and sum over them
    unused_qubits = list(set(range(qubit_count)) - set(targets))
    as_tensor = probabilities.reshape([2] * qubit_count)
    marginal = np.apply_over_axes(np.sum, as_tensor, unused_qubits).flatten()

    # Reorder qubits to match targets
    perm = _get_target_permutation(targets)
    return marginal[perm]


def partial_trace(
    density_matrix: np.ndarray,
    targets: Optional[list[int]] = None,
) -> np.ndarray:
    """Returns the reduced density matrix for the target qubits.

    If no target qubits are supplied, this method returns the trace of the density matrix.

    Args:
        density_matrix (np.ndarray): The density matrix to reduce,
            as a tensor product of qubit states.
        targets (list[int]): The qubits of the output reduced density matrix;
            if no target qubits are supplied, this method returns the trace of the density matrix.

    Returns:
        np.ndarray: The partial trace of the density matrix.
    """
    qubit_count = len(density_matrix.shape) // 2
    target_set = set(targets) if targets else set()
    nkeep = 2 ** len(target_set)
    idx1 = [i for i in range(qubit_count)]
    idx2 = [qubit_count + i if i in target_set else i for i in range(qubit_count)]
    tr_rho = np.einsum(density_matrix, idx1 + idx2).reshape(nkeep, nkeep)

    # reorder qubits to match target
    if targets:
        perm = _get_target_permutation(targets)
        tr_rho = tr_rho[:, perm]
        tr_rho = tr_rho[perm]
    return tr_rho


def _get_target_permutation(targets: Sequence[int]) -> Sequence[int]:
    """
    Return a permutation to reorder qubits to match targets
    """
    basis_states = np.array(list(itertools.product([0, 1], repeat=len(targets))))
    return np.ravel_multi_index(
        basis_states[:, np.argsort(np.argsort(targets))].T, [2] * len(targets)
    )
