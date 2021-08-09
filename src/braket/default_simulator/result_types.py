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

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from braket.ir import jaqcd

from braket.default_simulator.density_matrix_simulation import DensityMatrixSimulation
from braket.default_simulator.linalg_utils import marginal_probability, partial_trace
from braket.default_simulator.observables import (
    Hadamard,
    Hermitian,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    TensorProduct,
)
from braket.default_simulator.operation import Observable
from braket.default_simulator.operation_helpers import ir_matrix_to_ndarray
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.state_vector_simulation import StateVectorSimulation


def from_braket_result_type(result_type) -> ResultType:
    """Creates a `ResultType` corresponding to the given Braket instruction.

    Args:
        result_type: Result type for a circuit specified using the `braket.ir.jacqd` format.

    Returns:
        ResultType: Instance of specific `ResultType` corresponding to the type of result_type

    Raises:
        TypeError: If no concrete `ResultType` class has been registered
            for the Braket instruction type
    """
    return _from_braket_result_type(result_type)


@singledispatch
def _from_braket_result_type(result_type):
    raise TypeError(f"Result type {result_type} not recognized")


class ResultType(ABC):
    """
    An abstract class that when implemented defines a calculation on a
    quantum state simulation.
    """

    @abstractmethod
    def calculate(self, simulation: StateVectorSimulation) -> Any:
        # Return type of any due to lack of sum type support in Python
        """Calculate a result from the given quantum state vector simulation.

        Args:
            simulation (StateVectorSimulation): The quantum state vector simulation
                to use in the calculation

        Returns:
            Any: The result of the calculation
        """


class ObservableResultType(ResultType, ABC):
    """
    Holds an observable to perform a calculation in conjunction with a state.
    """

    def __init__(self, observable: Observable):
        """
        Args:
            observable (Observable): The observable for which the desired result is calculated
        """
        self._observable = observable

    @property
    def observable(self):
        """Observable: The observable for which the desired result is calculated."""
        return self._observable

    def calculate(self, simulation: Simulation) -> Union[float, List[float]]:
        """

        Args:
            simulation:

        Returns:

        """
        if self._observable.targets:
            return self._calculate_for_qubit(simulation, self._observable.apply)
        return [
            self._calculate_for_qubit(
                simulation, lambda state: self._observable.apply_to_qubit(state, qubit)
            )
            for qubit in range(simulation.qubit_count)
        ]

    @staticmethod
    @abstractmethod
    def _calculate_for_qubit(
        simulation: Simulation, apply_func: Callable[[np.ndarray], np.ndarray]
    ):
        """

        Args:
            simulation:
            apply_func:

        Returns:

        """
        raise NotImplementedError("")


class StateVector(ResultType):
    """
    Simply returns the given state vector.
    """

    def calculate(self, simulation: StateVectorSimulation) -> np.ndarray:
        """Return the given state vector of the simulation.

        Args:
            simulation (StateVectorSimulation): The simulation whose state vector will be returned

        Returns:
            np.ndarray: The state vector (before observables) of the simulation
        """
        return simulation.state_vector


@_from_braket_result_type.register
def _(_: jaqcd.StateVector):
    return StateVector()


class DensityMatrix(ResultType):
    """
    Simply returns the given density matrix.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        """
        Args:
            targets (Optional[List[int]]): The qubit indices on which the reduced density matrix
                are desired. If no targets are specified, the full density matrix are calculated.
                Default: `None`
        """
        self._targets = targets

    def calculate(self, simulation: DensityMatrixSimulation) -> np.ndarray:
        """Return the given density matrix of the simulation.

        Args:
            simulation (DensityMatrixSimulation): The simulation whose (full or reduced)
                density matrix will be returned.

        Returns:
            np.ndarray: The density matrix (before observables) of the simulation
        """
        if self._targets is None or len(self._targets) == simulation.qubit_count:
            return simulation.density_matrix
        else:
            if not all(ta in list(range(simulation.qubit_count)) for ta in self._targets):
                raise IndexError(
                    "Input target qubits must be within the range of the qubits in the circuit."
                )

            return partial_trace(simulation.density_matrix, simulation.qubit_count, self._targets)


@_from_braket_result_type.register
def _(densitymatrix: jaqcd.DensityMatrix):
    return DensityMatrix(densitymatrix.targets)


class Amplitude(ResultType):
    """
    Extracts the amplitudes of the desired computational basis states.
    """

    def __init__(self, states: List[str]):
        """
        Args:
            states (List[str]): The computational basis states whose amplitudes are desired
        """
        self._states = states

    def calculate(self, simulation: StateVectorSimulation) -> Dict[str, complex]:
        """Return the amplitudes of the desired computational basis states in the state
        of the given simulation.

        Args:
            simulation (StateVectorSimulation): The simulation whose state vector amplitudes
                will be returned

        Returns:
            Dict[str, complex]: A dict keyed on computational basis states as bitstrings,
            with corresponding values the amplitudes
        """
        state = simulation.state_vector
        return {basis_state: state[int(basis_state, 2)] for basis_state in self._states}


@_from_braket_result_type.register
def _(amplitude: jaqcd.Amplitude):
    return Amplitude(amplitude.states)


class Probability(ResultType):
    """
    Computes the marginal probabilities of computational basis states on the desired qubits.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        """
        Args:
            targets (Optional[List[int]]): The qubit indices on which probabilities are desired.
                If no targets are specified, the probabilities are calculated on the entire state.
                Default: `None`
        """
        self._targets = targets

    def calculate(self, simulation: StateVectorSimulation) -> np.ndarray:
        """Return the marginal probabilities of computational basis states on the target qubits.

        Probabilities are marginalized over all non-target qubits.

        Args:
            simulation (StateVectorSimulation): The simulation from which probabilities
                are calculated

        Returns:
            np.ndarray: An array of probabilities of length equal to 2^(number of target qubits),
            indexed by the decimal encoding of the computational basis state on the target qubits

        """
        return marginal_probability(
            simulation.probabilities,
            simulation.qubit_count,
            self._targets,
        )


@_from_braket_result_type.register
def _(probability: jaqcd.Probability):
    return Probability(probability.targets)


class Expectation(ObservableResultType):
    """
    Holds an observable :math:`O` to calculate its expected value.
    """

    def __init__(self, observable: Observable):
        """
        Args:
            observable (Observable): The observable for which expected value is calculated
        """
        super().__init__(observable)

    @staticmethod
    def _calculate_for_qubit(
        simulation: Simulation, apply_func: Callable[[np.ndarray], np.ndarray]
    ):
        return simulation.expectation(apply_func(simulation.state_as_tensor))


@_from_braket_result_type.register
def _(expectation: jaqcd.Expectation):
    return Expectation(_from_braket_observable(expectation.observable, expectation.targets))


class Variance(ObservableResultType):
    """
    Holds an observable :math:`O` to calculate its variance.
    """

    def __init__(self, observable: Observable):
        """
        Args:
            observable (Observable): The observable for which variance is calculated
        """
        super().__init__(observable)

    @staticmethod
    def _calculate_for_qubit(
        simulation: Simulation, apply_func: Callable[[np.ndarray], np.ndarray]
    ):
        squared = apply_func(apply_func(simulation.state_as_tensor))
        expectation = simulation.expectation(apply_func(simulation.state_as_tensor))
        return simulation.expectation(squared) - expectation ** 2


@_from_braket_result_type.register
def _(variance: jaqcd.Variance):
    return Variance(_from_braket_observable(variance.observable, variance.targets))


def _from_braket_observable(
    ir_observable: List[Union[str, List[List[List[float]]]]], ir_targets: Optional[List[int]] = None
) -> Observable:
    targets = list(ir_targets) if ir_targets else None
    if len(ir_observable) == 1:
        return _from_single_observable(ir_observable[0], targets)
    else:
        observable = TensorProduct(
            [_from_single_observable(factor, targets, is_factor=True) for factor in ir_observable]
        )
        if targets:
            raise ValueError(
                f"Found {len(targets)} more target qubits than the tensor product acts on"
            )
        return observable


def _from_single_observable(
    observable: Union[str, List[List[List[float]]]],
    targets: Optional[List[int]] = None,
    # IR tensor product observables are decoupled from targets
    is_factor: bool = False,
) -> Observable:
    if observable == "i":
        return Identity(_actual_targets(targets, 1, is_factor))
    elif observable == "h":
        return Hadamard(_actual_targets(targets, 1, is_factor))
    elif observable == "x":
        return PauliX(_actual_targets(targets, 1, is_factor))
    elif observable == "y":
        return PauliY(_actual_targets(targets, 1, is_factor))
    elif observable == "z":
        return PauliZ(_actual_targets(targets, 1, is_factor))
    else:
        try:
            matrix = ir_matrix_to_ndarray(observable)
            if is_factor:
                num_qubits = int(np.log2(len(matrix)))
                return Hermitian(matrix, _actual_targets(targets, num_qubits, True))
            else:
                return Hermitian(matrix, targets)
        except Exception:
            raise ValueError(f"Invalid observable specified: {observable}")


def _actual_targets(targets: List[int], num_qubits: int, is_factor: bool):
    if not is_factor:
        return targets
    try:
        return [targets.pop(0) for _ in range(num_qubits)]
    except Exception:
        raise ValueError("Insufficient qubits for tensor product")


@singledispatch
def _expectation(simulation, state_with_observable: np.ndarray) -> float:
    raise TypeError("")


@_expectation.register
def _(simulation: StateVectorSimulation, state_with_observable: np.ndarray):
    return float(
        np.dot(
            simulation.state_vector.conj(),
            np.reshape(state_with_observable, 2 ** len(state_with_observable.shape)),
        )
    )


@_expectation.register
def _(simulation: DensityMatrixSimulation, state_with_observable: np.ndarray):
    pass
