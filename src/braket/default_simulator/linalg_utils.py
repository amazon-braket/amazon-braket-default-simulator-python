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
from typing import List, Optional, Sequence, Tuple

import numpy as np


def multiply_matrix(state: np.ndarray, matrix: np.ndarray, targets: Tuple[int, ...]) -> np.ndarray:
    """Multiplies the given matrix by the given state, applying the matrix on the target qubits.

    Args:
        state (np.ndarray): The state to multiply the matrix by.
        matrix (np.ndarray): The matrix to apply to the state.
        targets (Tuple[int]): The qubits to apply the state on.

    Returns:
        np.ndarray: The state after the matrix has been applied.
    """
    gate_matrix = np.reshape(matrix, [2] * len(targets) * 2)
    axes = (
        np.arange(len(targets), 2 * len(targets)),
        targets,
    )
    product = np.tensordot(gate_matrix, state, axes=axes)

    # Axes given in `operation.targets` are in the first positions.
    unused_idxs = [idx for idx in range(len(state.shape)) if idx not in targets]
    permutation = list(targets) + unused_idxs
    # Invert the permutation to put the indices in the correct place
    inverse_permutation = np.argsort(permutation)
    return np.transpose(product, inverse_permutation)


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
        targets (List[int]): The qubits of the marginal distribution;
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
    targets: Optional[List[int]] = None,
) -> np.ndarray:
    """Returns the reduced density matrix for the target qubits.

    If no target qubits are supplied, this method returns the trace of the density matrix.

    Args:
        density_matrix (np.ndarray): The density matrix to reduce,
            as a tensor product of qubit states.
        targets (List[int]): The qubits of the output reduced density matrix;
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


def controlled_unitary(unitary: np.ndarray, negctrl: bool = False) -> np.ndarray:
    """
    Transform unitary matrix into a controlled unitary matrix.

    Args:
        unitary (np.ndarray): Unitary matrix operation.
        negctrl (bool): Whether to control the operation on the |0⟩ state,
            instead of the |1⟩ state.

    Returns:
        np.ndarray: A controlled version of the provided unitary matrix.
    """
    upper_left, bottom_right = np.eye(unitary.shape[0]), unitary
    if negctrl:
        upper_left, bottom_right = bottom_right, upper_left
    return np.block(
        [
            [upper_left, np.zeros_like(unitary)],
            [np.zeros_like(unitary), bottom_right],
        ]
    )


def measurement_sample(prob, target_count) -> Tuple[int]:
    basis_states = np.array(list(itertools.product([0, 1], repeat=target_count)))
    outcome_idx = np.random.choice(list(range(2**target_count)), p=prob)
    return tuple(basis_states[outcome_idx])


def measurement_collapse_dm(dm_tensor, targets, outcomes):
    """
    This needs to be modified to not delete qubits
    """
    # move the target qubit to the front of axes
    qubit_count = int(np.log2(dm_tensor.shape[0]))
    unused_idxs = [idx for idx in range(qubit_count) if idx not in targets]
    unused_idxs = [p+i*qubit_count for i in range(2) for p in unused_idxs] # convert indices to dm form
    target_indx = [p+i*qubit_count for i in range(2) for p in targets] # convert indices to dm form
    permutation = target_indx + unused_idxs
    inverse_permutation = np.argsort(permutation)

    # collapse the density matrix based on measuremnt outcome
    outcomes = tuple(i for _ in range(2) for i in outcomes)
    new_dm_tensor = np.zeros_like(dm_tensor)
    new_dm_tensor[outcomes] = np.transpose(dm_tensor, permutation)[outcomes]
    new_dm_tensor = np.transpose(new_dm_tensor, inverse_permutation)

    # normalize
    new_trace = np.trace(np.reshape(new_dm_tensor, (2**qubit_count, 2**qubit_count)))
    new_dm_tensor = new_dm_tensor / new_trace
    return new_dm_tensor


def measurement_collapse_sv(state_vector, targets, outcome):
    qubit_count = int(np.log2(state_vector.size))
    state_tensor = state_vector.reshape([2] * qubit_count)
    for qubit, measurement in zip(targets, outcome):
        state_tensor[(slice(None),) * qubit + (int(not measurement),)] = 0

    state_tensor /= np.linalg.norm(state_tensor)
    return state_tensor.flatten()

