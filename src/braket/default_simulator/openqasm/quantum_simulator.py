from typing import Optional, Sequence, Union

import numpy as np

from braket.default_simulator.linalg_utils import marginal_probability, multiply_matrix


class QuantumSimulator:
    """
    Qubits are initially in a ground state.
    Qubit indexing is as follows: |0123...n> (same as BDK, opposite from Qiskit)

    Currently, measurements and reset instructions are sampled, disrupting a pure state
    for a shots=0 analytic simulation. This is a temporary implementation for development
    of the interpreter/simulator while it stands alone and isn't connected to the bdk.
    Once it is more fleshed out, the interpreter will add validation to ensure that for
    a shots=0 simulation, no qubit is reused after being measured and no qubits are reset
    after being used.

    Eventually, a version of the OpenQASM simulator will be built with a density matrix
    quantum simulator. This version will support mid-circuit measurements and reset
    instructions for shots=0, as well as Braket noise pragmas. However, classical values
    that are assigned measurement outcomes cannot be used in further computation.
    """

    def __init__(self):
        self._num_qubits = 0
        self._state_tensor = np.array([], dtype=complex)

    @property
    def num_qubits(self):
        """number of qubits"""
        return self._num_qubits

    @property
    def state_vector(self):
        """state vector of shape (2 ** num_qubits,)"""
        return self._state_tensor.flatten()

    @property
    def probabilities(self):
        """probabilities of shape (2 ** num_qubits,)"""
        return np.abs(self.state_vector) ** 2

    @property
    def state_tensor(self):
        """state tensor of shape (2, 2,..., 2) with a dimension for each qubit"""
        return self._state_tensor.flatten()

    def add_qubits(self, num_qubits: int):
        """
        Allocate additional qubits with an initial state of |0⟩.

        Args:
            num_qubits (int): The number of additional qubits to allocate.
        """
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
        """normalize state"""
        self._state_tensor /= np.linalg.norm(self._state_tensor)

    def _flip_qubit(self, qubit: int):
        """
        Equivalent to applying an X gate to the qubit.

        Args:
            qubit (int): qubit to be flipped.
        """
        self._state_tensor = np.roll(
            self._state_tensor,
            1,
            axis=qubit,
        )

    def resolve_target(self, target: Optional[Union[int, Sequence[int]]] = None):
        """
        Converts target into a sequence, defaulting no target to all qubits
        and converting an int target into a sequence of length one.

        Args:
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).
        """
        if target is None:
            target = range(self.num_qubits)
        elif isinstance(target, int):
            target = (target,)
        return target

    def reset_qubits(self, target: Optional[Union[int, Sequence[int]]] = None):
        """
        reset one or more qubits

        This function emulates IBM's active reset, which measures the qubits
        and applies an X gate conditioned on the measurement outcomes.

        Args:
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).
        """
        target = self.resolve_target(target)
        measurement = self.measure_qubits(target)

        for qubit, result in zip(target, measurement):
            if result:
                self._flip_qubit(qubit)

    def measure_qubits(self, target: Union[int, Sequence[int]] = None):
        """
        Measure target qubits and update state vector.

        Args:
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).
        """
        # process input
        target = self.resolve_target(target)

        # sample state
        measured_state = self._sample_quantum_state(target)

        # zero out incompatible states
        for qubit, measurement in zip(target, measured_state):
            self._state_tensor[(slice(None),) * qubit + (int(not measurement),)] = 0

        # normalize state vector
        self._normalize_state()
        return measured_state

    def _sample_quantum_state(
        self, target: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Measure target qubit(s).

        Args:
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).

        Returns:
            np.ndarray: Boolean array of measurement outcomes.
        """
        target = self.resolve_target(target)
        marginal = marginal_probability(self.probabilities, target)
        sample = np.random.choice(2 ** len(target), p=marginal)
        return np.array([x == "1" for x in np.binary_repr(sample, len(target))])

    def execute_unitary(
        self, unitary: np.ndarray, target: Optional[Union[int, Sequence[int]]] = None
    ):
        """
        Execute unitary operation on provided target.

        Args:
            unitary (np.ndarray): Unitary matrix operation.
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).
        """
        target = self.resolve_target(target)
        self._state_tensor = multiply_matrix(self._state_tensor, unitary, target)

    def apply_phase(self, phase: float, target: Optional[Union[int, Sequence[int]]] = None):
        """
        Apply global phase shift to qubit target.

        Args:
            phase (float): Phase value.
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).
        """
        target = self.resolve_target(target)
        tensorized_target = self._get_tensorized_indices(target)
        self._state_tensor[tensorized_target] *= np.exp(phase * 1j)

    def _get_tensorized_indices(
        self, target: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Map indices in the state vector to indices in the state tensor.

        Args:
            target (Optional[Union[int, Sequence[int]]]): Qubit target.
                Default: range(self.num_qubits).

        Returns:
            np.ndarray: An array of multi-dimensional indices corresponding to the
                specified elements' locations in the state tensor.
        """
        target = self.resolve_target(target)
        return np.stack(np.unravel_index(target, self.state_tensor.shape), axis=-1)

    @staticmethod
    def generate_u(theta, phi, lambda_):
        """
        Built-in parameterized unitary defined by the OpenQASM 3 language specification.
        """
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lambda_)) * np.cos(theta / 2),
                ],
            ]
        )
