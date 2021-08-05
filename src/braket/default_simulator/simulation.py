# Copyright 2019-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from braket.default_simulator.operation import GateOperation


class Simulation(ABC):
    """
    This class tracks the evolution of a quantum system with `qubit_count` qubits.
    The state of system the evolves by application of `GateOperation`s using the `evolve()` method.
    """

    def __init__(self, qubit_count: int, shots: int):
        r"""
        Args:
            qubit_count (int): The number of qubits being simulated.
                All the qubits start in the :math:`\ket{\mathbf{0}}` computational basis state.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as density matrix
                or expectation, are generated.
        """
        self._qubit_count = qubit_count
        self._shots = shots

    @property
    def qubit_count(self) -> int:
        """int: The number of qubits being simulated by the simulation."""
        return self._qubit_count

    @property
    def shots(self) -> int:
        """
        int: The number of samples to take from the simulation.

        0 means no samples are taken, and results that require sampling
        to calculate cannot be returned.
        """
        return self._shots

    @abstractmethod
    def evolve(self, operations: List[GateOperation]) -> None:
        """Evolves the state of the simulation under the action of
        the specified gate operations.

        Args:
            operations (List[GateOperation]): Gate operations to apply for
                evolving the state of the simulation.

        Note:
            This method mutates the state of the simulation.
        """
        raise NotImplementedError("evolve is not implemented.")

    @property
    @abstractmethod
    def state_as_tensor(self) -> np.ndarray:
        """np.ndarray: The state of the simulation as a tensor product of qubit states."""
        raise NotImplementedError("")

    @abstractmethod
    def expectation(self, with_observables: np.ndarray) -> float:
        """The expected value of the observable applied to the state.

        Args:
            with_observables (np.ndarray): The state vector with the observable applied

        Returns:
            float: The expectated value of the observable.
        """
        raise NotImplementedError("")

    @abstractmethod
    def retrieve_samples(self) -> List[int]:
        """Retrieves samples of states from the state of the simulation,
        based on the probabilities.

        Returns:
            List[int]: List of states sampled according to their probabilities
            in the state. Each integer represents the decimal encoding of the
            corresponding computational basis state.
        """
        raise NotImplementedError("")

    @property
    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """np.ndarray: The probabilities of each computational basis state."""
        raise NotImplementedError("probabilities is not implemented.")
