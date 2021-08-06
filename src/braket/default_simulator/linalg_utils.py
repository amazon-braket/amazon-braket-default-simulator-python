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
from typing import List, Tuple, Optional

import numpy as np


def multiply_matrix(
        state: np.ndarray, matrix: np.ndarray, targets: Tuple[int, ...]
) -> np.ndarray:
    """ Multiplies

    Args:
        state (np.ndarray):
        matrix (np.ndarray):
        targets (Tuple[int]):

    Returns:
        np.ndarray:
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
    qubit_count: int,
    targets: List[int] = None,
) -> np.ndarray:
    """Return the marginal probability of the computational basis states.

    The marginal probability is obtained by summing the probabilities on
    the unused qubits. If no targets are specified, then the probability
    of all basis states is returned.

    Args:
        probabilities (np.ndarray):
        qubit_count (int):
        targets (List[int]):

    Returns:
        np.ndarray:
    """

    if targets is None or targets == list(range(qubit_count)):
        # All qubits targeted, no need to marginalize
        return probabilities

    targets = np.hstack(targets)

    # Find unused qubits and sum over them
    unused_qubits = list(set(range(qubit_count)) - set(targets))
    as_tensor = probabilities.reshape([2] * qubit_count)
    marginal = np.apply_over_axes(np.sum, as_tensor, unused_qubits).flatten()

    # Reorder qubits to match targets
    basis_states = np.array(list(itertools.product([0, 1], repeat=len(targets))))
    perm = np.ravel_multi_index(
        basis_states[:, np.argsort(np.argsort(targets))].T, [2] * len(targets)
    )
    return marginal[perm]


def partial_trace(
    density_matrix: np.ndarray,
    qubit_count: int,
    targets: Optional[List[int]] = None,
) -> np.ndarray:
    """ Returns the reduced density matrix for the target qubits.

    Args:
        density_matrix (np.ndarray):
        qubit_count (int):
        targets (List[int]):

    Returns:
        np.ndarray:
    """
    target_set = set(targets) if targets else set()
    nkeep = 2 ** len(target_set)
    idx1 = [i for i in range(qubit_count)]
    idx2 = [qubit_count + i if i in target_set else i for i in list(range(qubit_count))]
    tr_rho = density_matrix.reshape(np.array([2] * 2 * qubit_count))
    tr_rho = np.einsum(tr_rho, idx1 + idx2)
    return tr_rho.reshape(nkeep, nkeep)
