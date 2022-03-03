from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from openqasm3.ast import Expression, IntegerLiteral

from braket.default_simulator.linalg_utils import multiply_matrix


@dataclass
class QubitType:
    size: Optional[Expression]


@dataclass
class Qubit:
    state: np.ndarray

    def __init__(self, size: IntegerLiteral = None):
        size = size.value if size is not None else 1
        self.state = np.full((size, 2), np.nan, dtype=complex)

    def reset(self):
        self.state[:] = (1, 0)


class QuantumSimulator:
    """
    Qubits are initially in a ground state.
    Qubit indexing is as follows: |n...3210>
    """

    GROUND_STATE = (1, 0)

    def __init__(self):
        self._num_qubits = 0
        self._state_vector = np.array([], dtype=complex)

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def state_vector(self):
        return self._state_vector

    def add_qubits(self, num_qubits: int):
        """allocate additional qubits"""
        new_qubits = np.zeros(2 ** num_qubits, dtype=complex)
        new_qubits[0] = 1
        if self.num_qubits:
            self._state_vector = np.kron(self._state_vector, new_qubits)
        else:
            self._state_vector = new_qubits
        self._num_qubits += num_qubits

    def _normalize_state(self):
        self._state_vector /= np.linalg.norm(self._state_vector)

    def _tensorize_state(self):
        self._state_vector = self._state_vector.reshape(np.full(self.num_qubits, 2))

    def _detensorize_state(self):
        self._state_vector = self._state_vector.flatten()

    def reset_qubits(self, target: Union[int, Sequence]):
        """reset one or more qubits"""
        self.measure_qubits(target, 0)

    def measure_qubits(
        self, target: Union[int, Sequence], state: Optional[Union[bool, int, Sequence]] = None
    ):
        """
        Measure target qubits and update state vector. If state parameter is given,
        the outcome of the measurement will be that state, otherwise the measurement
        will be probabilistically sampled from the current quantum state.

        The state vector will be updated with all incompatible states being assigned
        a value of zero and then renormalized. In the case where the state vector has
        a magnitude of zero after the first step (the user specifies a measurement that
        has a probability of zero), the resulting state vector will be a uniform distribution
        of phase 0 over all compatible states and a warning will be raised.
        """
        # process input
        target = [target] if isinstance(target, int) else target
        state = QuantumSimulator._translate_state_to_array(state, len(target))
        measured_state = state if state is not None else self._sample_quantum_state(target)

        # get mask of states compatible with measurement
        mask = np.ones_like(self._state_vector, dtype=bool)
        for qubit, measurement in zip(target, measured_state):
            qubit_mask = self._get_qubit_mask(qubit)
            if not measurement:
                qubit_mask = ~qubit_mask
            mask &= qubit_mask

        # update state vector
        self._state_vector[~mask] = 0
        self._normalize_state()

    def _sample_quantum_state(self, target: Union[int, Sequence]) -> np.ndarray:
        """measure target qubit(s)"""
        raise NotImplementedError

    @staticmethod
    def _translate_state_to_array(
        state: Union[bool, int, Sequence], target_size: Optional[int] = None
    ):
        if isinstance(state, (bool, int)):
            return np.full(target_size, state, dtype=bool)
        else:
            if target_size != len(state):
                raise ValueError(f"Invalid state value {state} for target size {target_size}")
            return np.array(state)

    def _get_qubit_mask(self, qubit: int):
        """
        Get a mask of all the states where the given qubit is excited
        """
        return (np.arange(2 ** self.num_qubits) & (1 << qubit)).astype(bool)

    @staticmethod
    def _is_binary(x: int):
        return str(x)[:] == "0b"

    def execute_u(self, qubit, theta, phi, lambda_):
        self._tensorize_state()
        unitary = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lambda_)) * np.cos(theta / 2),
                ],
            ]
        )
        self._state_vector = multiply_matrix(self._state_vector, unitary, (qubit,))
        self._detensorize_state()
