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

from string import ascii_lowercase, ascii_uppercase
from typing import List, Tuple

import numpy as np
import opt_einsum

from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation import Simulation


class UnitaryMatrixSimulation(Simulation):
    """
    This class tracks the evolution of the unitary matrix of a quantum system with
    `qubit_count` qubits. The unitary of system evolves by applications of `GateOperation`s
    using the `evolve()` method.
    """

    def __init__(self, qubit_count: int, shots: int):
        """
        Args:
            qubit_count (int): The number of qubits being simulated.
            shots (int): The number of samples to take from the simulation.
                0 is the only valid value for this simulator.
        """
        super().__init__(qubit_count=qubit_count, shots=shots)
        initial_unitary = np.eye(2 ** qubit_count, dtype=complex)
        self._unitary_matrix = initial_unitary

    def evolve(self, operations: List[GateOperation]) -> None:
        """Evolves the unitary of the simulation under the action of
        the specified gate operations.

        Args:
            operations (List[GateOperation]): Gate operations to apply for
                evolving the unitary of the simulation.

        Note:
            This method mutates the state of the simulation.
        """
        self._unitary_matrix = UnitaryMatrixSimulation._apply_operations(
            self._unitary_matrix, self._qubit_count, operations
        )

    @staticmethod
    def _apply_operations(
        unitary: np.ndarray, qubit_count: int, operations: List[GateOperation]
    ) -> np.ndarray:
        un_tensor = np.reshape(unitary, qubit_count * [2, 2])
        for operation in operations:
            matrix = operation.matrix
            targets = operation.targets
            un_tensor = UnitaryMatrixSimulation._apply_operation(
                un_tensor, qubit_count, matrix, targets
            )

        return np.reshape(un_tensor, 2 * [2 ** qubit_count])

    @staticmethod
    def _apply_operation(
        un_tensor: np.ndarray, qubit_count: int, matrix: np.ndarray, targets: Tuple[int, ...]
    ) -> np.ndarray:
        subscripts = UnitaryMatrixSimulation._einsum_subscripts(targets, qubit_count)

        target_count = len(targets)
        gate_matrix = np.reshape(matrix, target_count * [2, 2])

        return opt_einsum.contract(subscripts, gate_matrix, un_tensor, dtype=complex, casting="no")

    @staticmethod
    def _einsum_subscripts(targets: Tuple[int, ...], qubit_count: int) -> str:
        target_count = len(targets)
        assert target_count + qubit_count <= len(ascii_lowercase), (
            "Simulation qubit count + gate target count "
            f"can not be bigger than {len(ascii_lowercase)}"
        )

        un_left_labels = ascii_lowercase[:qubit_count]
        un_right_labels = ascii_uppercase[:qubit_count]

        gate_left_labels = ascii_lowercase[: -1 - target_count : -1]
        gate_right_labels = "".join(un_left_labels[-1 - target] for target in reversed(targets))

        result_left_labels = list(un_left_labels)
        for pos, target in enumerate(targets):
            result_left_labels[-1 - target] = gate_left_labels[-1 - pos]
        result_left_labels = "".join(result_left_labels)

        return (
            f"{gate_left_labels}{gate_right_labels}, "
            f"{un_left_labels}{un_right_labels} -> "
            f"{result_left_labels}{un_right_labels}"
        )

    @property
    def unitary_matrix(self) -> np.ndarray:
        """
        np.ndarray: The unitary matrix equivalent to the entire simulation.
        """
        return self._unitary_matrix
