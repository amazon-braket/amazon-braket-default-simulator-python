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
import os
from collections.abc import Sequence
from typing import Optional

import numba as nb
import numpy as np

os.environ["NUMBA_OPT"] = "3"
os.environ["NUMBA_CPU_NAME"] = "native"

_NEG_CONTROL_SLICE = slice(None, 1)
_CONTROL_SLICE = slice(1, None)
_NO_CONTROL_SLICE = slice(None, None)

BASIS_MAPPING = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

_QUBIT_THRESHOLD = nb.int32(10)


class QuantumGateDispatcher:
    def __init__(self, n_qubits: int):
        """
        Makes a way to dispatch to different optimized functions based on qubit count.
        """
        self.n_qubits = n_qubits
        self.use_large = n_qubits > _QUBIT_THRESHOLD

        if self.use_large:
            self.apply_single_qubit_gate = _apply_single_qubit_gate_large
            self.apply_swap = _apply_swap_large
            self.apply_controlled_phase_shift = _apply_controlled_phase_shift_large
            self.apply_cnot = _apply_cnot_large
            self.apply_two_qubit_gate = _apply_two_qubit_gate_large
        else:
            self.apply_single_qubit_gate = _apply_single_qubit_gate_small
            self.apply_swap = _apply_swap_small
            self.apply_controlled_phase_shift = _apply_controlled_phase_shift_small
            self.apply_cnot = _apply_cnot_small
            self.apply_two_qubit_gate = _apply_two_qubit_gate_small


def multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: Optional[tuple[int, ...]] = (),
    control_state: Optional[tuple[int, ...]] = (),
    out: Optional[np.ndarray] = None,
    dispatcher: Optional[QuantumGateDispatcher] = None,
    return_swap_info: bool = False,
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
        return_swap_info (bool): For backwards comp. Used to indicate whether the ping-pong buffer swaps should happen.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    if dispatcher is None:
        dispatcher = QuantumGateDispatcher(state.size)

    if out is None:
        out = np.zeros_like(state, dtype=complex)

    if not controls:
        out, swap = _multiply_matrix(state, matrix, targets, out, dispatcher)
        if return_swap_info:
            return out, swap
        else:
            return out

    control_state = control_state or (1,) * len(controls)

    ctrl_slices = [_NO_CONTROL_SLICE] * len(state.shape)
    for i, state_val in zip(controls, control_state):
        ctrl_slices[i] = _NEG_CONTROL_SLICE if state_val == 0 else _CONTROL_SLICE
    ctrl_tuple = tuple(ctrl_slices)

    np.copyto(out, state)

    controlled_slice = out[ctrl_tuple]

    _, swap = _multiply_matrix(state[ctrl_tuple], matrix, targets, controlled_slice, dispatcher)

    if return_swap_info:
        return out, swap
    else:
        return out


def _apply_single_qubit_gate_small(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies single gates using array slicing."""
    shape = state.shape
    before_size = int(np.prod(shape[:target]))
    after_size = int(np.prod(shape[target + 1 :]))

    state_reshaped = state.reshape(before_size, 2, after_size)
    out_reshaped = out.reshape(before_size, 2, after_size)

    state_0 = state_reshaped[:, 0, :]
    state_1 = state_reshaped[:, 1, :]

    a, b, c, d = matrix.flat
    out_reshaped[:, 0, :] = a * state_0 + b * state_1
    out_reshaped[:, 1, :] = c * state_0 + d * state_1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_single_qubit_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies single gates using bit masking."""
    a, b, c, d = matrix.flat
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state.flat[idx0]
        state1 = state.flat[idx1]

        out.flat[idx0] = a * state0 + b * state1
        out.flat[idx1] = c * state0 + d * state1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_cnot_large(
    state: np.ndarray, control: int, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:  # pragma: no cover
    """CNOT optimization path with numba."""
    n_qubits = state.ndim
    total_size = state.size
    iterations = total_size >> 2
    if control > target:
        target, control = control, target
    target_bit_pos = n_qubits - target - 1
    control_bit_pos = n_qubits - control - 1

    control_stride = 1 << control_bit_pos
    target_stride = 1 << target_bit_pos

    target_jump = target_stride if target_bit_pos != n_qubits - 1 else 0
    control_jump = control_stride if control_bit_pos != n_qubits - 1 else 0

    should_target_jump = target_jump or 1
    should_control_jump = control_jump or 1
    if control_bit_pos - target_bit_pos >= (n_qubits - target_bit_pos) // 2:
        should_control_jump = max(should_control_jump // 2, 1)

    # when the control qubits are off by 1, there seems to be a "super" jump at each of the target_stride lengths
    if control_bit_pos - target_bit_pos == 1:
        combined_jump = target_jump + control_jump
        for i in nb.prange(iterations):
            idx0 = control_stride + i + (i // should_target_jump) * combined_jump
            idx1 = idx0 + target_stride

            temp = state.flat[idx0]
            state.flat[idx0] = state.flat[idx1]
            state.flat[idx1] = temp
    else:
        for i in nb.prange(iterations):
            idx0 = (
                control_stride
                + i
                + (i // should_target_jump) * target_jump
                + (i // should_control_jump) * control_jump
            )
            idx1 = idx0 + target_stride

            temp = state.flat[idx0]
            state.flat[idx0] = state.flat[idx1]
            state.flat[idx1] = temp
    return state, False


def _apply_cnot_small(
    state: np.ndarray, control: int, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """CNOT optimization path."""
    n_qubits = state.ndim

    slice_list = [slice(None)] * n_qubits

    slice_list[control] = 1
    slice_list[target] = 0
    slices_c1t0 = tuple(slice_list)

    slice_list[target] = 1
    slices_c1t1 = tuple(slice_list)

    temp = np.copy(state[slices_c1t0])
    state[slices_c1t0] = state[slices_c1t1]
    state[slices_c1t1] = temp

    return state, False


def _apply_swap_small(
    state: np.ndarray, qubit_0: int, qubit_1: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Swap gate implementation using numpy's swapaxes."""
    np.copyto(out, np.swapaxes(state, qubit_0, qubit_1))
    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_swap_large(
    state: np.ndarray, qubit_0: int, qubit_1: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:  # pragma: no cover
    """Swap gate implementation using bit manipulation."""
    n_qubits = state.ndim
    total_size = 1 << n_qubits

    mask_0 = 1 << (n_qubits - 1 - qubit_0)
    mask_1 = 1 << (n_qubits - 1 - qubit_1)

    for i in nb.prange(total_size):
        bit_0 = (i & mask_0) != 0
        bit_1 = (i & mask_1) != 0

        if bit_0 != bit_1:
            j = i ^ mask_0 ^ mask_1

            if i < j:
                state.flat[i], state.flat[j] = state.flat[j], state.flat[i]

    return state, False


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_controlled_phase_shift_large(  # pragma: no cover
    state: np.ndarray, phase_factor: float, controls: np.ndarray, target: int
) -> tuple[np.ndarray, bool]:
    """C Phase shift gate optimization path for larger vectors using bit masks.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        phase_factor (float): The multiplier based on the gate's angle.
        controls (np.ndarray): List of control gates.
        target (int): The qubit to apply the state on.
        out (np.ndarray): Output array for result.

    Returns:
        np.ndarray: The state array with the controlled phase shift gate applied.
    """
    n_qubits = state.ndim
    shift_base = n_qubits - 1
    mask = 1 << (shift_base - target)

    for c in controls:
        mask |= 1 << (shift_base - c)

    # here, we search for the lowest bit that is 0 and set that as our step size
    # this iterates through slightly too many values so we need a check in the for loop.
    step = ~mask & (mask + 1)

    num_free_bits = n_qubits - len(controls) - 1
    total_valid_indices = 1 << num_free_bits

    mask_bits_above_step = mask & ~((step << 1) - 1)
    next_carry_point = mask_bits_above_step & (-mask_bits_above_step)
    chunk_size = next_carry_point // step
    chunk_mask = chunk_size - 1
    two_minor_jump = next_carry_point << 1

    # if there are jumps, we look to tile remaining 2 * 2 ^ (qubits - len(controls) - 1) items and check those.
    # this is an approximation that avoids specifically creating the indices needed.
    if (
        mask_bits_above_step > 0
        and two_minor_jump + mask < state.size
        and chunk_size != total_valid_indices
    ):
        chunk_shift = 0
        temp = chunk_size
        while temp > 1:
            temp >>= 1
            chunk_shift += 1

        max_chunk_num = (state.size - mask - step * chunk_mask) // two_minor_jump
        max_safe_iterations = (max_chunk_num + 1) * chunk_size
        iterations = min(total_valid_indices << 1, max_safe_iterations)

        for i in nb.prange(iterations):
            idx = mask + step * (i & chunk_mask) + (i >> chunk_shift) * two_minor_jump
            if (idx & mask) == mask:
                state.flat[idx] *= phase_factor
    else:
        for i in nb.prange(total_valid_indices):
            idx = mask + step * i
            state.flat[idx] *= phase_factor

    return state, False


def _apply_controlled_phase_shift_small(
    state: np.ndarray, phase_factor: float, controls, target: int
) -> tuple[np.ndarray, bool]:
    """C Phase shift gate optimization path for smaller vectors using numpy slicing."""
    slices = [slice(None)] * len(state.shape)
    for c in controls:
        slices[c] = 1
    slices[target] = 1

    state[tuple(slices)] *= phase_factor

    return state, False


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_two_qubit_gate_large(
    state: np.ndarray,
    matrix: np.ndarray,
    target0: int,
    target1: int,
    out: np.ndarray,
) -> tuple[np.ndarray, bool]:  # pragma: no cover
    """Two-qubit gate implementation using bit manipulation."""
    n_qubits = state.ndim
    total_size = 1 << n_qubits

    mask_0 = 1 << (n_qubits - 1 - target0)
    mask_1 = 1 << (n_qubits - 1 - target1)

    for i in nb.prange(total_size):
        out_basis = ((i & mask_0) != 0) * 2 + ((i & mask_1) != 0)

        base = i & ~(mask_0 | mask_1)

        result = (
            matrix[out_basis, 0] * state.flat[base]
            + matrix[out_basis, 1] * state.flat[base | mask_1]
            + matrix[out_basis, 2] * state.flat[base | mask_0]
            + matrix[out_basis, 3] * state.flat[base | mask_0 | mask_1]
        )

        out.flat[i] = result

    return out, True


def _apply_two_qubit_gate_small(
    state: np.ndarray,
    matrix: np.ndarray,
    target0: int,
    target1: int,
    out: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Two qubit gate application with numppy."""
    n_qubits = state.ndim
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

    return out, True


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

    threshold = 1e-10
    diag = np.diag(matrix)

    if (
        abs(matrix[2, 3] - 1) < threshold
        and abs(matrix[3, 2] - 1) < threshold
        and abs(diag[0] - 1) < threshold
        and abs(diag[1] - 1) < threshold
    ):
        return dispatcher.apply_cnot(state, target0, target1, out)
    elif matrix[1, 2] == 1 and matrix[2, 1] == 1:
        if diag[0] == 1 and diag[3] == 1:  # pragma: no cover
            return dispatcher.apply_swap(state, target0, target1, out)
    elif abs(diag[0] - 1) < threshold and abs(diag[1] - 1) < threshold:
        if abs(diag[2] - 1) < threshold:
            angle = np.angle(diag[3])
            phase_factor = np.exp(1j * angle)
            if abs(diag[3] - phase_factor) < threshold:  # pragma: no cover
                return dispatcher.apply_controlled_phase_shift(
                    state, phase_factor, (target0,), target1
                )
    return dispatcher.apply_two_qubit_gate(state, matrix, target0, target1, out)


def _apply_single_qubit_gate(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies single gates based on qubit count.
    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        target (int): The qubit to apply the state on.
        out (np.ndarray): Output array to store result in.
    Returns:
        np.ndarray: Modified state vector
    """
    n_qubits = state.ndim

    if n_qubits > _QUBIT_THRESHOLD:
        return _apply_single_qubit_gate_large(state, matrix, target, out)
    else:
        return _apply_single_qubit_gate_small(state, matrix, target, out)


def _multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    out: np.ndarray,
    dispatcher: QuantumGateDispatcher,
) -> tuple[np.ndarray, bool]:
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
        return _apply_single_qubit_gate(state, matrix, targets[0], out)
    elif len(targets) == 2:
        return _apply_two_qubit_gate(state, matrix, targets, out, dispatcher)

    num_targets = len(targets)
    gate_matrix = np.reshape(matrix, [2] * num_targets * 2)
    axes = (np.arange(num_targets, 2 * num_targets), targets)

    product = np.tensordot(gate_matrix, state, axes=axes)
    unused_idxs = [idx for idx in range(state.ndim) if idx not in targets]

    np.copyto(out, np.transpose(product, np.argsort([*targets, *unused_idxs])))
    return out, True


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
