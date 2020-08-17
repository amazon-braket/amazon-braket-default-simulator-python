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

from typing import List, Tuple, Union

import numpy as np
from braket.default_simulator.operation import GateOperation, KrausOperation, Observable
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.operation_helpers import get_matrix
from braket.default_simulator import densitymatrix_simulation_helpers


class DensityMatrixSimulation(Simulation):
    """
    This class tracks the evolution of the density matrix of a quantum system with
    `qubit_count` qubits. The state of system evolves by applications of `GateOperations`
    and `KrausOperations` cusing the `evolve()` method.
    """

    def __init__(self, qubit_count: int, shots: int):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as density matrix
                or expectation, are generated.
        """
        super().__init__(qubit_count = qubit_count, shots=shots)
        initial_state = np.zeros((2 ** qubit_count, 2 ** qubit_count), dtype=complex)
        initial_state[0, 0] = 1
        self._density_matrix = initial_state
        self._post_observables = None

    def evolve(self, operations: List[Union[GateOperation, KrausOperation]]) -> None:
        """ Evolves the state of the simulation under the action of the specified gate
        and noise operations.

        Args:
            operations (List[Union[GateOperation, KrausOperation]]): Operations to apply for
                evolving the state of the simulation.

        Note:
            This method mutates the state of the simulation.
        """
        self._density_matrix = DensityMatrixSimulation._apply_operations(
            self._density_matrix, self._qubit_count, operations
        )

    def apply_observables(self, observables: List[Observable]) -> None:
        """ Applies the diagonalizing matrices of the given observables
        to the state of the simulation.

        This method can only be called once.

        Args:
            observables (List[Observable]): The observables to apply

        Raises:
            RuntimeError: If this method is called more than once
        """
        if self._post_observables is not None:
            raise RuntimeError("Observables have already been applied.")
        self._post_observables = DensityMatrixSimulation._apply_operations(
            self._density_matrix, self._qubit_count, observables
        )

    @staticmethod
    def _apply_operations(
        state: np.ndarray, qubit_count: int, operations: List[Union[GateOperation, KrausOperation]]
    ) -> np.ndarray:
        """ Applies the gate and noise operations to the density matrix.

        Args:
            state (np.array): initial density matrix
            qubit_count (int): number of qubit in the circuit
            operations (List[Union[GateOperation, KrausOperation]]): list of GateOperation and
                KrausOperation to be applied to the density matrix

        Returns:
            state (np.array): output density matrix
        """
        return densitymatrix_simulation_helpers._apply_operations(state, qubit_count, operations)


    def retrieve_samples(self) -> List[int]:
        """ Retrieves samples of states from the density matrix of the simulation,
        based on the probabilities.

        Returns:
            List[int]: List of states sampled according to their probabilities
            in the density matrix. Each integer represents the decimal encoding of the
            corresponding computational basis state.
        """
        return np.random.choice(self._density_matrix.shape[0], p=self.probabilities, size=self._shots)

    @property
    def state(self) -> np.ndarray:
        """
        np.ndarray: The current state of the simulation.
        """
        return self.density_matrix

    @property
    def density_matrix(self) -> np.ndarray:
        """
        np.ndarray: The density matrix specifying the current state of the simulation.
        """
        return self._density_matrix

    @property
    def state_with_observables(self) -> np.ndarray:
        """
        np.ndarray: The final density matrix of the simulation after application of observables.

        Raises:
            RuntimeError: If observables have not been applied
        """
        if self._post_observables is None:
            raise RuntimeError("No observables applied")
        return self._post_observables

    @property
    def probabilities(self) -> np.ndarray:
        """np.ndarray: The probabilities of each computational basis state."""
        return self.probabilities_from_state(self._density_matrix)

    @staticmethod
    def probabilities_from_state(state) -> np.array:
        """np.ndarray: The probabilities of each computational basis state of a
        given state.
        """
        return np.real(np.diag(state))
