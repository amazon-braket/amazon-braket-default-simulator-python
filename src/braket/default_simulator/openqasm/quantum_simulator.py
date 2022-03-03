from typing import Optional, Sequence, Union

import numpy as np

from braket.default_simulator.linalg_utils import multiply_matrix


class QuantumSimulator:
    """
    Qubits are initially in a ground state.
    Qubit indexing is as follows: |0123...n>
    """

    GROUND_STATE = (1, 0)

    def __init__(self):
        self._num_qubits = 0
        self._state_tensor = np.array([], dtype=complex)

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def state_vector(self):
        return self._state_tensor.flatten()

    @property
    def state_tensor(self):
        return self._state_tensor.flatten()

    def add_qubits(self, num_qubits: int):
        """allocate additional qubits"""
        if not self._num_qubits:
            self._state_tensor = np.zeros(np.full(num_qubits, 2))
            self._state_tensor[(0,) * num_qubits] = 1
        else:
            for _ in range(num_qubits):
                self._state_tensor = np.stack(
                    (self._state_tensor, np.zeros_like(self._state_tensor)),
                    axis=-1,
                )
        self._num_qubits += num_qubits

    def _normalize_state(self):
        self._state_tensor /= np.linalg.norm(self._state_tensor)

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

        # zero out incompatible states
        for qubit, measurement in zip(target, measured_state):
            self._state_tensor[(slice(None),) * qubit + (int(not measurement),)] = 0

        # normalize state vector
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

    def execute_unitary(self, unitary, target: Union[int, Sequence[int]]):
        if isinstance(target, int):
            target = (target,)
        self._state_tensor = multiply_matrix(self._state_tensor, unitary, target)

    @staticmethod
    def generate_u(theta, phi, lambda_):
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lambda_)) * np.cos(theta / 2),
                ],
            ]
        )
