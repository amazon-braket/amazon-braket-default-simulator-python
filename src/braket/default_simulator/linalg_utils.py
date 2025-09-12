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
from typing import Union

import numba as nb
import numpy as np

nb.config.NUMBA_OPT = 3
# CPU settings
nb.config.NUMBA_CPU_NAME = "native"
nb.config.NUMBA_SLP_VECTORIZE = 1
# threading settings
nb.config.THREADING_LAYER = "workqueue"
nb.config.NUMBA_NUM_THREADS = -1
# Logging settings (TODO is add a way to turn these on for dev work)
nb.config.NUMBA_DEBUG = 0
nb.config.WARNINGS = False
nb.config.CAPTURED_ERRORS = "new_style"

_NEG_CONTROL_SLICE = slice(None, 1)
_CONTROL_SLICE = slice(1, None)
_NO_CONTROL_SLICE = slice(None, None)

BASIS_MAPPING = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

_QUBIT_THRESHOLD = nb.int32(10)


class QuantumGateDispatcher:
    def __init__(self, n_qubits: int):
        """
        Dispatcher for performance-optimized implementations of quantum gates.  It automatically
        selects between small-circuit (NumPy-based) and large-circuit (Numba JIT-compiled) 
        implementations based on the number of qubits in a circuit. 
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
    controls: tuple[int, ...] | None = (),
    control_state: tuple[int, ...] | None = (),
    out: np.ndarray | None = None,
    dispatcher: QuantumGateDispatcher | None = None,
    return_swap_info: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, bool]]:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits,
    controlling the operation as specified.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        controls (tuple[int, ...] | None): The qubits to control the operation on. Default ().
        control_state (tuple[int, ...] | None): A tuple of same length as `controls` with either
            a 0 or 1 in each index, corresponding to whether to control on the `|0⟩` or `|1⟩` state.
            Default (1,) * len(controls).
        out (np.ndarray | None): Preallocated result array to reduce overhead of creating a new array each time.
        dispatcher(QuantumGateDispatcher): Dispatch to optimized functions based on qubit
            count.
        return_swap_info (bool): For backwards comp. Used to indicate whether the ping-pong buffer swaps should happen.

    Returns:
        Union[np.ndarray, tuple[np.ndarray, bool]]: The state after the matrix has been applied.
            When return_swap_info is True, returns a tuple of (state, swap_occurred).
            When return_swap_info is False, returns just the state array.
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

    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    out_reshaped[:, 0, :] = a * state_0 + b * state_1
    out_reshaped[:, 1, :] = c * state_0 + d * state_1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_single_qubit_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies single gates using bit masking."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n = state.ndim - target - 1
    mask = (np.int64(1) << n) - 1

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | (np.int64(1) << n)

        s0, s1 = state_flat[idx0], state_flat[idx1]

        out_flat[idx0] = a * s0 + b * s1
        out_flat[idx1] = c * s0 + d * s1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_x_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies X gate using bit masking.

    Matrix: [[0, 1],
             [1, 0]]
    """
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        out_flat[idx0] = state_flat[idx1]
        out_flat[idx1] = state_flat[idx0]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_z_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies Z gate using bit masking.

    Matrix: [[1,  0],
             [0, -1]]
    """

    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = -state_flat[i]
        else:
            out_flat[i] = state_flat[i]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_y_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies Y gate using bit masking.

    Matrix: [[0, -1j],
             [1j,  0]]
    """
    b, c = matrix[0, 1], matrix[1, 0]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state_flat[idx0]
        state1 = state_flat[idx1]

        out_flat[idx0] = b * state1
        out_flat[idx1] = c * state0

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_s_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies S gate using bit masking.

    Matrix: [[1, 0],
             [0, 1j]]
    """
    d = matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = state_flat[i]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_si_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies S† gate using bit masking.

    Matrix: [[1,  0],
             [0, -1j]]
    """
    d = matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = state_flat[i]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_t_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies T gate using bit masking.

    Matrix: [[1, 0],
             [0, exp(1j*π/4)]]
    """
    d = matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = state_flat[i]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_ti_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies T† gate using bit masking.

    Matrix: [[1, 0],
             [0, exp(-1j*π/4)]]
    """
    d = matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = state_flat[i]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_hadamard_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies Hadamard gate using bit masking with hardcoded values.

    Matrix: [[1,  1],
             [1, -1]] / √2
    """
    val = matrix[0, 0]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state_flat[idx0]
        state1 = state_flat[idx1]

        out_flat[idx0] = val * (state0 + state1)
        out_flat[idx1] = val * (state0 - state1)

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_v_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies V gate (sqrt(X)) using bit masking.

    Matrix: [[0.5+0.5j, 0.5-0.5j],
             [0.5-0.5j, 0.5+0.5j]]
    """
    a, b = matrix[0, 0], matrix[0, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state_flat[idx0]
        state1 = state_flat[idx1]

        out_flat[idx0] = a * state0 + b * state1
        out_flat[idx1] = b * state0 + a * state1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_vi_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies V† gate (sqrt(X)†) using bit masking.

    Matrix: [[0.5-0.5j, 0.5+0.5j],
             [0.5+0.5j, 0.5-0.5j]]
    """
    a, b = matrix[0, 0], matrix[0, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state_flat[idx0]
        state1 = state_flat[idx1]

        out_flat[idx0] = a * state0 + b * state1
        out_flat[idx1] = b * state0 + a * state1

    return out, True


def _apply_rx_gate_small(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies RX rotation gate using array slicing."""
    shape = state.shape
    before_size = int(np.prod(shape[:target]))
    after_size = int(np.prod(shape[target + 1 :]))

    state_reshaped = state.reshape(before_size, 2, after_size)
    out_reshaped = out.reshape(before_size, 2, after_size)

    state_0 = state_reshaped[:, 0, :]
    state_1 = state_reshaped[:, 1, :]

    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    out_reshaped[:, 0, :] = a * state_0 + b * state_1
    out_reshaped[:, 1, :] = c * state_0 + d * state_1

    return out, True


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _apply_rx_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies RX rotation gate using bit masking.

    Matrix: [[cos(θ/2),    -i*sin(θ/2)],
             [-i*sin(θ/2), cos(θ/2)]]
    """
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state_flat[idx0]
        state1 = state_flat[idx1]

        out_flat[idx0] = a * state0 + b * state1
        out_flat[idx1] = c * state0 + d * state1

    return out, True


def _apply_ry_gate_small(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies RY rotation gate using array slicing."""
    shape = state.shape
    before_size = int(np.prod(shape[:target]))
    after_size = int(np.prod(shape[target + 1 :]))

    state_reshaped = state.reshape(before_size, 2, after_size)
    out_reshaped = out.reshape(before_size, 2, after_size)

    state_0 = state_reshaped[:, 0, :]
    state_1 = state_reshaped[:, 1, :]

    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    out_reshaped[:, 0, :] = a * state_0 + b * state_1
    out_reshaped[:, 1, :] = c * state_0 + d * state_1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_ry_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies RY rotation gate using bit masking.

    Matrix: [[cos(θ/2),  -sin(θ/2)],
             [sin(θ/2),   cos(θ/2)]]
    """
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)
    shifted_target_mask = np.int64(target_mask - 1)

    half_size = state.size >> 1
    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(half_size):
        idx0 = (i & ~(shifted_target_mask)) << 1 | (i & (shifted_target_mask))
        idx1 = idx0 | target_mask

        state0 = state_flat[idx0]
        state1 = state_flat[idx1]

        out_flat[idx0] = a * state0 + b * state1
        out_flat[idx1] = c * state0 + d * state1

    return out, True


def _apply_rz_gate_small(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies RZ rotation gate using array slicing."""
    shape = state.shape
    before_size = int(np.prod(shape[:target]))
    after_size = int(np.prod(shape[target + 1 :]))

    state_reshaped = state.reshape(before_size, 2, after_size)
    out_reshaped = out.reshape(before_size, 2, after_size)

    state_0 = state_reshaped[:, 0, :]
    state_1 = state_reshaped[:, 1, :]

    a, d = matrix[0, 0], matrix[1, 1]
    out_reshaped[:, 0, :] = a * state_0
    out_reshaped[:, 1, :] = d * state_1

    return out, True


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _apply_rz_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies RZ rotation gate using bit masking.

    Matrix: [[exp(-i*θ/2), 0],
             [0,           exp(i*θ/2)]]
    """
    a, d = matrix[0, 0], matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = a * state_flat[i]

    return out, True


def _apply_phase_shift_gate_small(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies phase shift gate using array slicing."""
    shape = state.shape
    before_size = int(np.prod(shape[:target]))
    after_size = int(np.prod(shape[target + 1 :]))

    state_reshaped = state.reshape(before_size, 2, after_size)
    out_reshaped = out.reshape(before_size, 2, after_size)

    state_0 = state_reshaped[:, 0, :]
    state_1 = state_reshaped[:, 1, :]

    a, d = matrix[0, 0], matrix[1, 1]
    out_reshaped[:, 0, :] = a * state_0
    out_reshaped[:, 1, :] = d * state_1

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_phase_shift_gate_large(  # pragma: no cover
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies phase shift gate using bit masking."""
    a, d = matrix[0, 0], matrix[1, 1]
    target_bit = state.ndim - target - 1
    target_mask = np.int64(1 << target_bit)

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(state.size):
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = a * state_flat[i]

    return out, True


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_cnot_large(
    state: np.ndarray, control: int, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:  # pragma: no cover
    """CNOT optimization path with numba."""
    n_qubits = state.ndim
    total_size = state.size
    iterations = total_size >> 2

    target_bit_pos = n_qubits - target - 1
    control_bit_pos = n_qubits - control - 1

    control_stride = 1 << control_bit_pos
    target_stride = 1 << target_bit_pos

    if control_bit_pos > target_bit_pos:
        larger_bit_pos = control_bit_pos
        smaller_bit_pos = target_bit_pos

        larger_jump = control_stride if control_bit_pos != n_qubits - 1 else 0
        smaller_jump = target_stride if target_bit_pos != n_qubits - 1 else 0

        swap_offset = target_stride
    else:
        larger_bit_pos = target_bit_pos
        smaller_bit_pos = control_bit_pos

        larger_jump = target_stride if target_bit_pos != n_qubits - 1 else 0
        smaller_jump = control_stride if control_bit_pos != n_qubits - 1 else 0

        swap_offset = target_stride

    should_smaller_jump = smaller_jump or 1
    should_larger_jump = larger_jump or 1

    if larger_bit_pos - smaller_bit_pos >= (n_qubits - smaller_bit_pos) // 2:
        should_larger_jump = max(should_larger_jump // 2, 1)

    state_flat = state.reshape(-1)

    if larger_bit_pos - smaller_bit_pos == 1:
        combined_jump = smaller_jump + larger_jump
        for i in nb.prange(iterations):
            idx0 = control_stride + i + (i // should_smaller_jump) * combined_jump
            idx1 = idx0 + swap_offset

            state_flat[idx0], state_flat[idx1] = state_flat[idx1], state_flat[idx0]
    else:
        for i in nb.prange(iterations):
            idx0 = (
                control_stride
                + i
                + (i // should_smaller_jump) * smaller_jump
                + (i // should_larger_jump) * larger_jump
            )
            idx1 = idx0 + swap_offset

            state_flat[idx0], state_flat[idx1] = state_flat[idx1], state_flat[idx0]

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
    iterations = total_size >> 2

    pos_0 = n_qubits - 1 - qubit_0
    pos_1 = n_qubits - 1 - qubit_1

    if pos_0 > pos_1:
        pos_0, pos_1 = pos_1, pos_0

    mask_0 = 1 << pos_0
    mask_1 = 1 << pos_1

    state_flat = state.reshape(-1)

    for i in nb.prange(iterations):
        base = i + ((i >> pos_0) << pos_0)
        base += (base >> pos_1) << pos_1

        idx0 = base | mask_1
        idx1 = base | mask_0

        state_flat[idx0], state_flat[idx1] = state_flat[idx1], state_flat[idx0]

    return state, False


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_controlled_phase_shift_large(  # pragma: no cover
    state: np.ndarray, phase_factor: complex, controls: np.ndarray, target: int
) -> tuple[np.ndarray, bool]:
    """C Phase shift gate optimization path for larger vectors using bit masks.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        phase_factor (complex): The multiplier based on the gate's angle.
        controls (np.ndarray): List of control gates.
        target (int): The qubit to apply the state on.

    Returns:
        tuple[np.ndarray, bool]: A tuple containing the state array with the controlled phase shift gate applied
            and a boolean indicating whether a buffer swap occurred.
    """
    n_qubits = state.ndim

    num_controlled_qubits = len(controls) + 1
    iterations = state.size >> num_controlled_qubits

    controlled_positions = np.empty(num_controlled_qubits, dtype=np.int64)
    controlled_positions[0] = np.int64(n_qubits - 1 - target)
    for i, control in enumerate(controls):
        controlled_positions[i + 1] = np.int64(n_qubits - 1 - control)
    controlled_positions.sort()

    controlled_mask = np.int64(0)
    for pos in controlled_positions:
        controlled_mask |= np.int64(1) << pos

    state_flat = state.reshape(-1)

    if len(controlled_positions) == 1:
        pos = controlled_positions[0]
        lower_mask = np.int64((1 << pos) - 1)
        upper_mask = ~lower_mask

        for i in nb.prange(iterations):
            idx = (i & lower_mask) | ((i & upper_mask) << 1)
            final_idx = idx | controlled_mask
            state_flat[final_idx] *= phase_factor

    elif len(controlled_positions) == 2:
        pos0 = controlled_positions[0]
        pos1 = controlled_positions[1]
        mask0 = np.int64((1 << pos0) - 1)
        mask1 = np.int64((1 << pos1) - 1)

        for i in nb.prange(iterations):
            idx = i
            idx = (idx & mask0) | ((idx & ~mask0) << 1)
            idx = (idx & mask1) | ((idx & ~mask1) << 1)
            final_idx = idx | controlled_mask
            state_flat[final_idx] *= phase_factor

    elif len(controlled_positions) == 3:
        pos0 = controlled_positions[0]
        pos1 = controlled_positions[1]
        pos2 = controlled_positions[2]
        mask0 = np.int64((1 << pos0) - 1)
        mask1 = np.int64((1 << pos1) - 1)
        mask2 = np.int64((1 << pos2) - 1)

        for i in nb.prange(iterations):
            idx = i
            idx = (idx & mask0) | ((idx & ~mask0) << 1)
            idx = (idx & mask1) | ((idx & ~mask1) << 1)
            idx = (idx & mask2) | ((idx & ~mask2) << 1)
            final_idx = idx | controlled_mask
            state_flat[final_idx] *= phase_factor

    else:
        masks = np.empty(len(controlled_positions), dtype=np.int64)
        for j, pos in enumerate(controlled_positions):
            masks[j] = np.int64((1 << pos) - 1)

        for i in nb.prange(iterations):
            idx = np.int64(i)
            for j in range(len(controlled_positions)):
                mask = masks[j]
                idx = (idx & mask) | ((idx & ~mask) << 1)
            final_idx = idx | controlled_mask
            state_flat[final_idx] *= phase_factor

    return state, False


def _apply_controlled_phase_shift_small(
    state: np.ndarray, phase_factor: complex, controls, target: int
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
    mask_both = mask_0 | mask_1

    state_flat = state.reshape(-1)
    out_flat = out.reshape(-1)

    for i in nb.prange(total_size):
        if (i & mask_both) == 0:
            s0 = state_flat[i]
            s1 = state_flat[i | mask_1]
            s2 = state_flat[i | mask_0]
            s3 = state_flat[i | mask_both]

            out_flat[i] = (
                matrix[0, 0] * s0 + matrix[0, 1] * s1 + matrix[0, 2] * s2 + matrix[0, 3] * s3
            )

            out_flat[i | mask_1] = (
                matrix[1, 0] * s0 + matrix[1, 1] * s1 + matrix[1, 2] * s2 + matrix[1, 3] * s3
            )

            out_flat[i | mask_0] = (
                matrix[2, 0] * s0 + matrix[2, 1] * s1 + matrix[2, 2] * s2 + matrix[2, 3] * s3
            )

            out_flat[i | mask_both] = (
                matrix[3, 0] * s0 + matrix[3, 1] * s1 + matrix[3, 2] * s2 + matrix[3, 3] * s3
            )

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
) -> tuple[np.ndarray, bool]:
    """Two-qubit gates optimization path.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        out (np.ndarray): Output array for result.
        dispatcher(QuantumGateDispatcher): Dispatch to optimized functions based on qubit
            count.

    Returns:
        tuple[np.ndarray, bool]: A tuple containing the state after the matrix has been applied
            and a boolean indicating whether a buffer swap occurred.

    """
    target0, target1 = targets
    threshold = 1e-10

    if (
        abs(matrix[0, 0] - 1) < threshold
        and abs(matrix[0, 1]) < threshold
        and abs(matrix[0, 2]) < threshold
        and abs(matrix[0, 3]) < threshold
        and abs(matrix[1, 0]) < threshold
        and abs(matrix[1, 1] - 1) < threshold
        and abs(matrix[1, 2]) < threshold
        and abs(matrix[1, 3]) < threshold
        and abs(matrix[2, 0]) < threshold
        and abs(matrix[2, 1]) < threshold
        and abs(matrix[2, 2]) < threshold
        and abs(matrix[2, 3] - 1) < threshold
        and abs(matrix[3, 0]) < threshold
        and abs(matrix[3, 1]) < threshold
        and abs(matrix[3, 2] - 1) < threshold
        and abs(matrix[3, 3]) < threshold
    ):
        return dispatcher.apply_cnot(state, target0, target1, out)

    if (
        abs(matrix[0, 0] - 1) < threshold
        and abs(matrix[0, 1]) < threshold
        and abs(matrix[0, 2]) < threshold
        and abs(matrix[0, 3]) < threshold
        and abs(matrix[1, 0]) < threshold
        and abs(matrix[1, 1]) < threshold
        and abs(matrix[1, 2] - 1) < threshold
        and abs(matrix[1, 3]) < threshold
        and abs(matrix[2, 0]) < threshold
        and abs(matrix[2, 1] - 1) < threshold
        and abs(matrix[2, 2]) < threshold
        and abs(matrix[2, 3]) < threshold
        and abs(matrix[3, 0]) < threshold
        and abs(matrix[3, 1]) < threshold
        and abs(matrix[3, 2]) < threshold
        and abs(matrix[3, 3] - 1) < threshold
    ):
        return dispatcher.apply_swap(state, target0, target1, out)

    if (
        abs(matrix[0, 0] - 1) < threshold
        and abs(matrix[1, 1] - 1) < threshold
        and abs(matrix[2, 2] - 1) < threshold
        and abs(matrix[0, 1]) < threshold
        and abs(matrix[0, 2]) < threshold
        and abs(matrix[0, 3]) < threshold
        and abs(matrix[1, 0]) < threshold
        and abs(matrix[1, 2]) < threshold
        and abs(matrix[1, 3]) < threshold
        and abs(matrix[2, 0]) < threshold
        and abs(matrix[2, 1]) < threshold
        and abs(matrix[2, 3]) < threshold
        and abs(matrix[3, 0]) < threshold
        and abs(matrix[3, 1]) < threshold
        and abs(matrix[3, 2]) < threshold
    ):
        phase_factor = matrix[3, 3]
        return dispatcher.apply_controlled_phase_shift(state, phase_factor, (target0,), target1)

    return dispatcher.apply_two_qubit_gate(state, matrix, target0, target1, out)


def _apply_single_qubit_gate(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Applies single gates based on qubit count and gate type.

    For large qubit counts, dispatches to optimized implementations for common gates.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        target (int): The qubit to apply the state on.
        out (np.ndarray): Output array to store result in.
    Returns:
        tuple[np.ndarray, bool]: A tuple containing the modified state vector and a boolean
            indicating whether a buffer swap occurred.
    """
    n_qubits = state.ndim

    # TODO: This needs to be cleaned up. Planning on adding gate identifiers for faster dispatching.
    if n_qubits > _QUBIT_THRESHOLD:
        if (
            abs(matrix[0, 0]) < 1e-12
            and abs(matrix[0, 1] - 1) < 1e-12
            and abs(matrix[1, 0] - 1) < 1e-12
            and abs(matrix[1, 1]) < 1e-12
        ):
            return _apply_x_gate_large(state, matrix, target, out)
        elif (
            abs(matrix[0, 0] - 1) < 1e-12
            and abs(matrix[0, 1]) < 1e-12
            and abs(matrix[1, 0]) < 1e-12
            and abs(matrix[1, 1] + 1) < 1e-12
        ):
            return _apply_z_gate_large(state, matrix, target, out)
        elif (
            abs(matrix[0, 0]) < 1e-12
            and abs(matrix[0, 1] + 1j) < 1e-12
            and abs(matrix[1, 0] - 1j) < 1e-12
            and abs(matrix[1, 1]) < 1e-12
        ):
            return _apply_y_gate_large(state, matrix, target, out)
        elif (
            abs(matrix[0, 0] - 0.7071067811865476) < 1e-12
            and abs(matrix[0, 1] - 0.7071067811865476) < 1e-12
            and abs(matrix[1, 0] - 0.7071067811865476) < 1e-12
            and abs(matrix[1, 1] + 0.7071067811865476) < 1e-12
        ):
            return _apply_hadamard_gate_large(state, matrix, target, out)
        elif abs(matrix[0, 1]) < 1e-12 and abs(matrix[1, 0]) < 1e-12:
            if (
                abs(matrix[0, 0] - 1) < 1e-12
                and abs(matrix[0, 1]) < 1e-12
                and abs(matrix[1, 0]) < 1e-12
                and abs(matrix[1, 1] - 1j) < 1e-12
            ):
                return _apply_s_gate_large(state, matrix, target, out)
            elif (
                abs(matrix[0, 0] - 1) < 1e-12
                and abs(matrix[0, 1]) < 1e-12
                and abs(matrix[1, 0]) < 1e-12
                and abs(matrix[1, 1] + 1j) < 1e-12
            ):
                return _apply_si_gate_large(state, matrix, target, out)
            elif (
                abs(matrix[0, 0] - 1) < 1e-12
                and abs(matrix[0, 1]) < 1e-12
                and abs(matrix[1, 0]) < 1e-12
                and abs(matrix[1, 1] - (0.7071067811865476 + 0.7071067811865475j)) < 1e-12
            ):
                return _apply_t_gate_large(state, matrix, target, out)
            elif (
                abs(matrix[0, 0] - 1) < 1e-12
                and abs(matrix[0, 1]) < 1e-12
                and abs(matrix[1, 0]) < 1e-12
                and abs(matrix[1, 1] - (0.7071067811865476 - 0.7071067811865475j)) < 1e-12
            ):
                return _apply_ti_gate_large(state, matrix, target, out)
            elif matrix[0, 0] == 1 and matrix[1, 1] != 1:
                return _apply_phase_shift_gate_large(state, matrix, target, out)
            else:
                return _apply_rz_gate_large(state, matrix, target, out)
        elif (
            abs(matrix[0, 0] - (0.5 + 0.5j)) < 1e-12
            and abs(matrix[0, 1] - (0.5 - 0.5j)) < 1e-12
            and abs(matrix[1, 0] - (0.5 - 0.5j)) < 1e-12
            and abs(matrix[1, 1] - (0.5 + 0.5j)) < 1e-12
        ):
            return _apply_v_gate_large(state, matrix, target, out)
        elif (
            abs(matrix[0, 0] - (0.5 - 0.5j)) < 1e-12
            and abs(matrix[0, 1] - (0.5 + 0.5j)) < 1e-12
            and abs(matrix[1, 0] - (0.5 + 0.5j)) < 1e-12
            and abs(matrix[1, 1] - (0.5 - 0.5j)) < 1e-12
        ):
            return _apply_vi_gate_large(state, matrix, target, out)
        else:
            return _apply_single_qubit_gate_large(state, matrix, target, out)
    else:
        if (
            abs(matrix[0, 0]) < 1e-12
            and abs(matrix[0, 1] - 1) < 1e-12
            and abs(matrix[1, 0] - 1) < 1e-12
            and abs(matrix[1, 1]) < 1e-12
        ):
            return _apply_single_qubit_gate_small(state, matrix, target, out)
        elif abs(matrix[0, 1]) < 1e-12 and abs(matrix[1, 0]) < 1e-12:
            if matrix[0, 0] == 1 and matrix[1, 1] != 1:
                return _apply_phase_shift_gate_small(state, matrix, target, out)
            else:
                return _apply_rz_gate_small(state, matrix, target, out)
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
        tuple[np.ndarray, bool]: A tuple containing the state after the matrix has been applied
            and a boolean indicating whether a buffer swap occurred.
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
    targets: list[int] | None = None,
) -> np.ndarray:
    """Returns the reduced density matrix for the target qubits.

    If no target qubits are supplied, this method returns the trace of the density matrix.

    Args:
        density_matrix (np.ndarray): The density matrix to reduce,
            as a tensor product of qubit states.
        targets (list[int] | None): The qubits of the output reduced density matrix;
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
