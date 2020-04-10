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

import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from braket.default_simulator.operation import Observable


class ResultType(ABC):
    """
    An abstract class that when implemented defines a calculation on a quantum state.
    """

    @abstractmethod
    def calculate(self, state: np.ndarray) -> Any:  # Python doesn't support sum types
        """ Calculate a result from the given quantum state.

        Args:
            state (np.ndarray): The quantum state vector to use in the calculation

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


class StateVector(ResultType):
    """
    Simply returns the given state vector.
    """

    def calculate(self, state: np.ndarray) -> np.ndarray:
        """ Return the given state vector.

        Args:
            state (np.ndarray): The state vector to retrieve

        Returns:
            np.ndarray: The state vector itself
        """
        return state


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

    def calculate(self, state: np.ndarray) -> Dict[str, complex]:
        """ Return the amplitudes of the desired computational basis states in the given state.

        Args:
            state (np.ndarray): The state vector from which amplitudes are extracted

        Returns:
            Dict[str, complex]: A dict keyed on computational basis states as bitstrings,
            with corresponding values the amplitudes
        """
        return {basis_state: state[int(basis_state, 2)] for basis_state in self._states}


class Probability(ResultType):
    """
    Computes the marginal probabilities of computational basis states on the desired qubits.
    """

    def __init__(self, targets: List[int]):
        """
        Args:
            targets (List[int]): The qubit indices on which probabilities are desired
        """
        self._targets = targets

    def calculate(self, state: np.ndarray) -> np.ndarray:
        """ Return the marginal probabilities of computational basis states on the target qubits.

        Probabilities are marginalized over all non-target qubits.

        Args:
            state (np.ndarray): The state vector from which probabilities are calculated

        Returns:
            np.ndarray: An array of probabilities of length equal to 2^(number of target qubits),
            indexed by the decimal encoding of the computational basis state on the target qubits

        """
        return _marginal_probability(state, self._targets)


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

    def calculate(self, state: np.ndarray) -> np.ndarray:
        r""" Computes the expected value of :math:`O` in the given state.

        The expected value of the observable :math:`O` in a state :math:`\ket{\psi}`
        is defined as

        .. math::

            \expectation{O}{\psi} = \bra{\psi} O \ket{\psi}

        Args:
            state (np.ndarray): The state that the expected value of the observable
                will be calculated in

        Returns:
            np.ndarray: The expected value of the observable :math:`O` in the given state
        """
        prob = _marginal_probability(state, self._observable.targets)
        eigenvalues = self._observable.eigenvalues
        return (prob @ eigenvalues).real


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

    def calculate(self, state: np.ndarray) -> np.ndarray:
        r""" Computes the variance of :math:`O` in the given state.

        The variance of the observable :math:`O` in a state :math:`\ket{\psi}`
        is defined from the expected value the same way it is in statistics:

        .. math::

            \variance{O}{\psi} = \expectation{O^2}{\psi} - \expectation{O}{\psi}^2

        Args:
            state (np.ndarray): The state that the variance will be calculated in

        Returns:
            np.ndarray: The variance of the observable :math:`O` in the given state
        """
        prob = _marginal_probability(state, self._observable.targets)
        eigenvalues = self._observable.eigenvalues
        return prob @ (eigenvalues ** 2) - (prob @ eigenvalues).real ** 2


class Sample(ObservableResultType):
    """
    Holds an observable :math:`O` to take samples from measuring it.
    """

    def __init__(self, observable: Observable, num_samples: int):
        """
        Args:
            observable (Observable): The observable to measure
            num_samples (int): The number of samples to take
        """
        super().__init__(observable)
        self._num_samples = num_samples

    def calculate(self, state: np.ndarray) -> np.ndarray:
        """ Takes samples from measuring :math:`O`.

        Measurements are taken in the eigenbasis of the observable,
        so they are the eigenvalues of the observable.

        Args:
            state (np.ndarray): The state vector to sample from

        Returns:
            np.ndarray: A list of measurements of the observable of length
            equal to the number of samples
        """
        targets = self._observable.targets
        prob = _marginal_probability(state, targets)
        return np.random.choice(self._observable.eigenvalues, p=prob, size=self._num_samples)


def _marginal_probability(state, targets=None) -> np.ndarray:
    """ Return the marginal probability of the computational basis states.

    The marginal probability is obtained by summing the probabilities on
    the unused qubits. If no targets are specified, then the probability
    of all basis states is returned.
    """

    num_qubits = int(np.log2(state.size))
    probabilities = np.abs(state) ** 2

    if targets is None or targets == list(range(num_qubits)):
        # All qubits targeted, no need to marginalize
        return probabilities

    targets = np.hstack(targets)

    # Find unused qubits and sum over them
    unused_qubits = list(set(range(num_qubits)) - set(targets))
    as_tensor = probabilities.reshape([2] * num_qubits)
    marginal = np.apply_over_axes(np.sum, as_tensor, unused_qubits).flatten()

    # Reorder qubits to match targets
    basis_states = np.array(list(itertools.product([0, 1], repeat=len(targets))))
    perm = np.ravel_multi_index(
        basis_states[:, np.argsort(np.argsort(targets))].T, [2] * len(targets)
    )
    return marginal[perm]
