# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import List

import numpy as np
import opt_einsum
from braket.default_simulator.operation import Operation
from braket.default_simulator.simulation_strategies.simulation_helpers import get_matrix


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: List[Operation], batch_size: int
) -> np.ndarray:
    """ Applies operations to a state vector in batches of size `batch_size`.

    The operation list is split into contiguous partitions of length `batch_size` (with remainder);
    within each partition, contraction order is optimized among the gates, and the partitions
    themselves are contracted in the order they appear. Larger partitions can be significantly
    faster, although this is not guaranteed, but will use more memory.

    Args:
        state (np.ndarray): The state vector to apply the given operations to, as a type
            (num_qubits, 0) tensor
        qubit_count (int): The number of qubits in the state
        operations (List[Operation]): The operations to apply to the state vector
        batch_size: The size of the partition of operations to contract

    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (num_qubits, 0) tensor
    """
    # TODO: Write algorithm to determine partition size based on operations and qubit count
    partitions = [operations[i : i + batch_size] for i in range(0, len(operations), batch_size)]

    for partition in partitions:
        state = _contract_operations(state, qubit_count, partition)

    return state


def _contract_operations(
    state: np.ndarray, qubit_count: int, operations: List[Operation]
) -> np.ndarray:
    contraction_parameters = [state, list(range(qubit_count))]
    index_substitutions = {i: i for i in range(qubit_count)}
    next_index = qubit_count
    for operation in operations:
        matrix = get_matrix(operation)
        targets = operation.targets

        # `operation` is not added tp the contraction parameters if
        # it is an observable with a trivial diagonalizing matrix
        if matrix is not None:
            if targets:
                # lower indices, which will be traced out
                covariant = [index_substitutions[i] for i in targets]

                # upper indices, which will replace the contracted indices in the state vector
                contravariant = list(range(next_index, next_index + len(covariant)))

                indices = contravariant + covariant
                # matrix as type-(len(contravariant), len(covariant)) tensor
                matrix_as_tensor = np.reshape(matrix, [2] * len(indices))

                contraction_parameters += [matrix_as_tensor, indices]
                next_index += len(covariant)
                index_substitutions.update(
                    {targets[i]: contravariant[i] for i in range(len(targets))}
                )
            else:
                # `operation` is an observable, and the only element in `operations`
                for qubit in range(qubit_count):
                    # Since observables don't overlap,
                    # there's no need to track index replacements
                    contraction_parameters += [matrix, [next_index, qubit]]
                    index_substitutions[qubit] = next_index
                    next_index += 1

    # Ensure state is in correct order
    new_indices = [index_substitutions[i] for i in range(qubit_count)]
    contraction_parameters.append(new_indices)
    return opt_einsum.contract(*contraction_parameters)
