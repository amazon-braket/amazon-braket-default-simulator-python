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
from functools import cache
from typing import Optional

import numpy as np


def multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: Optional[tuple[int, ...]] = (),
    control_state: Optional[tuple[int, ...]] = (),
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

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    if not controls:
        return _multiply_matrix(state, matrix, targets)

    control_state = control_state or (1,) * len(controls)

    ctrl_slices = [slice(None)] * len(state.shape)
    for i, state_val in zip(controls, control_state):
        ctrl_slices[i] = slice(None, 1) if state_val == 0 else slice(1, None)

    state[tuple(ctrl_slices)] = _multiply_matrix(state[tuple(ctrl_slices)], matrix, targets)
    return state


@cache
def _compute_inverse_permutation(targets, num_qubits):
    """Compute and cache the inverse permutation for a given target configuration."""
    unused_idxs = [idx for idx in range(num_qubits) if idx not in targets]
    return np.argsort([*targets, *unused_idxs])


def _multiply_matrix(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (tuple[int]): The qubits to apply the state on.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    num_targets = len(targets)

    gate_matrix = np.reshape(matrix, [2] * num_targets * 2)
    axes = (np.arange(num_targets, 2 * num_targets), targets)
    product = np.tensordot(gate_matrix, state, axes=axes)

    inverse_perm = _compute_inverse_permutation(tuple(targets), len(state.shape))
    return np.transpose(product, inverse_perm)


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
