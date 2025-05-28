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

import numpy as np

_NEG_CONTROL_SLICE = slice(None, 1)
_CONTROL_SLICE = slice(1, None)
_NO_CONTROL_SLICE = slice(None, None)

BASIS_MAPPING = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

# Preallocate for up to 42 qubits
_SLICE_NONE_ARRAYS_0 = {n: [slice(None)] * n for n in range(1, 43)}
_SLICE_NONE_ARRAYS_1 = {n: [slice(None)] * n for n in range(1, 43)}


def multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: Optional[tuple[int, ...]] = (),
    control_state: Optional[tuple[int, ...]] = (),
    out: Optional[np.ndarray] = None,
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

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """

    if out is None:
        out = np.zeros_like(state, dtype=complex)

    if not controls:
        return _multiply_matrix(state, matrix, targets, out)

    control_state = control_state or (1,) * len(controls)

    ctrl_slices = [_NO_CONTROL_SLICE] * len(state.shape)
    for i, state_val in zip(controls, control_state):
        ctrl_slices[i] = _NEG_CONTROL_SLICE if state_val == 0 else _CONTROL_SLICE
    ctrl_tuple = tuple(ctrl_slices)

    np.copyto(out, state)

    controlled_slice = out[ctrl_tuple]
    _multiply_matrix(state[ctrl_tuple], matrix, targets, controlled_slice)

    return out


def _apply_single_qubit_gate(
    state: np.ndarray, matrix: np.ndarray, target: int, out: np.ndarray
) -> np.ndarray:
    """Applies single gates using array slicing.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        target (int): The qubit to apply the state on.
        out (np.ndarray): Output array to store result in.

    Returns:
        np.ndarray: Modified state vector
    """
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]

    # Rather than allocate these arrays each time, preallocate these, update and reset them each time
    slices_0 = _SLICE_NONE_ARRAYS_0[len(state.shape)]
    slices_0[target] = 0
    slices_0_tuple = tuple(slices_0)

    slices_1 = _SLICE_NONE_ARRAYS_0[len(state.shape)]
    slices_1[target] = 1
    slices_1_tuple = tuple(slices_1)

    out[slices_0_tuple] = a * state[slices_0_tuple] + b * state[slices_1_tuple]
    out[slices_1_tuple] = c * state[slices_0_tuple] + d * state[slices_1_tuple]

    # Clean up step
    slices_0[target] = slice(None)
    slices_1[target] = slice(None)
    return out


def _apply_cnot(state: np.ndarray, control: int, target: int, out: np.ndarray) -> np.ndarray:
    """CNOT optimization path."""
    np.copyto(out, state)

    # Rather than allocate these arrays each time, preallocate these, update and reset them each time
    slices_c1t0 = _SLICE_NONE_ARRAYS_0[len(state.shape)]
    slices_c1t0[control] = 1
    slices_c1t0[target] = 0
    slices_c1t0_tuple = tuple(slices_c1t0)

    slices_c1t1 = _SLICE_NONE_ARRAYS_1[len(state.shape)]
    slices_c1t1[control] = 1
    slices_c1t1[target] = 1
    slices_c1t1_tuple = tuple(slices_c1t1)

    out[slices_c1t0_tuple] = out[slices_c1t1_tuple]
    out[slices_c1t1_tuple] = state[slices_c1t0_tuple]

    # Clean up step
    slices_c1t0[control] = slice(None)
    slices_c1t0[target] = slice(None)
    slices_c1t1[control] = slice(None)
    slices_c1t1[target] = slice(None)

    return out


def _apply_swap(state: np.ndarray, qubit_0: int, qubit_1: int, out: np.ndarray) -> np.ndarray:
    """Swap gate optimization path."""
    np.copyto(out, np.swapaxes(state, qubit_0, qubit_1))
    return out


def _apply_controlled_phase_shift(
    state: np.ndarray, angle: float, controls: tuple[int, ...], target: int, out: np.ndarray
) -> np.ndarray:
    """Controlled phase shift optimization path."""
    np.copyto(out, state)

    slices = _SLICE_NONE_ARRAYS_0[len(state.shape)]
    for c in controls:
        slices[c] = 1
    slices[target] = 1

    out[tuple(slices)] *= np.exp(1j * angle)

    # Clean up step
    for c in controls:
        slices[c] = slice(None)
    slices[target] = slice(None)

    return out


def _apply_two_qubit_gate(
    state: np.ndarray, matrix: np.ndarray, targets: tuple[int, int], out: np.ndarray
) -> np.ndarray:
    """Two-qubit gates optimization path.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        out (np.ndarray): Output array for result.

    Returns:
        np.ndarray: The state after the matrix has been applied.

    """
    target0, target1 = targets
    n_qubits = len(state.shape)

    if matrix.ndim != 2 or matrix.shape != (4, 4):
        matrix = matrix.reshape(4, 4)

    # Moving away from np.allclose here to avoid slightly more expensive checks
    diag = np.diag(matrix)
    angle = np.angle(matrix[3, 3])

    if (
        abs(diag[0] - 1) < 1e-10
        and abs(diag[1] - 1) < 1e-10
        and abs(diag[2] - 1) < 1e-10
        and abs(diag[3] - np.exp(1j * angle)) < 1e-10
    ):
        return _apply_controlled_phase_shift(state, angle, (target0,), target1, out)
    elif matrix[2, 3] == 1 and matrix[3, 2] == 1 and np.all(np.diag(matrix)[[0, 1]] == 1):
        return _apply_cnot(state, target0, target1, out)
    elif matrix[1, 2] == 1 and matrix[2, 1] == 1 and np.all(np.diag(matrix)[[0, 3]] == 1):
        return _apply_swap(state, target0, target1, out)

    # If there was a way around this, that would be great. Haven't figured one out yet.
    out.fill(0)

    # TODO: Make this global/one time computed
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
) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.
        out (np.ndarray): Output array for result.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    if len(targets) == 1:
        return _apply_single_qubit_gate(state, matrix, targets[0], out)
    elif len(targets) == 2:
        return _apply_two_qubit_gate(state, matrix, targets, out)

    num_targets = len(targets)
    gate_matrix = np.reshape(matrix, [2] * num_targets * 2)
    axes = (np.arange(num_targets, 2 * num_targets), targets)

    product = np.tensordot(gate_matrix, state, axes=axes)
    unused_idxs = [idx for idx in range(len(state.shape)) if idx not in targets]

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
