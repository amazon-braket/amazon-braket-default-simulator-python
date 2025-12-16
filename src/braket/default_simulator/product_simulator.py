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

"""
product_simulator.py

Efficient simulator for product state quantum circuits.

This module provides a simulator optimized for circuits that maintain
product state structure (no entanglement). Each qubit is tracked independently,
enabling O(n) simulation instead of O(2^n).

Mathematical Basis:
- Product states: |ψ⟩ = |ψ_0⟩ ⊗ |ψ_1⟩ ⊗ ... ⊗ |ψ_{n-1}⟩
- Single-qubit gates applied directly to individual qubit states
- Amplitudes computed as products of individual qubit amplitudes

Complexity:
- Space: O(n) for n qubits
- Gate application: O(1) per single-qubit gate
- Amplitude query: O(n)
- Sampling: O(shots * n)
- State vector: O(2^n) - only when explicitly requested

Dependencies:
- numpy
- numba (for acceleration)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation

_NUMBA_THRESHOLD = 10


@nb.njit(cache=True, fastmath=True)
def _compute_amplitude_numba(
    qubit_states: np.ndarray, outcome: int, n_qubits: int
) -> complex:  # pragma: no cover
    amp = 1.0 + 0j
    for k in range(n_qubits):
        bit = (outcome >> (n_qubits - 1 - k)) & 1
        amp *= qubit_states[k, bit]
    return amp


@nb.njit(parallel=True, cache=True, fastmath=True)
def _compute_state_vector_parallel(
    qubit_states: np.ndarray, n_qubits: int
) -> np.ndarray:  # pragma: no cover
    size = 1 << n_qubits
    state = np.zeros(size, dtype=np.complex128)
    for i in nb.prange(size):
        amp = 1.0 + 0j
        for k in range(n_qubits):
            bit = (i >> (n_qubits - 1 - k)) & 1
            amp *= qubit_states[k, bit]
        state[i] = amp
    return state


@nb.njit(parallel=True, cache=True, fastmath=True)
def _compute_probabilities_parallel(
    qubit_states: np.ndarray, n_qubits: int
) -> np.ndarray:  # pragma: no cover
    size = 1 << n_qubits
    probs = np.zeros(size, dtype=np.float64)
    for i in nb.prange(size):
        p = 1.0
        for k in range(n_qubits):
            bit = (i >> (n_qubits - 1 - k)) & 1
            p *= qubit_states[k, bit].real ** 2 + qubit_states[k, bit].imag ** 2
        probs[i] = p
    return probs


@nb.njit(cache=True)
def _sample_product_state(
    qubit_probs_0: np.ndarray, n_qubits: int, shots: int, random_values: np.ndarray
) -> np.ndarray:  # pragma: no cover
    results = np.zeros(shots, dtype=np.int64)
    for s in range(shots):
        outcome = 0
        for i in range(n_qubits):
            bit = 0 if random_values[s, i] < qubit_probs_0[i] else 1
            outcome = (outcome << 1) | bit
        results[s] = outcome
    return results


class ProductStateSimulator:
    """
    Simulator for quantum circuits that maintain product state structure.

    This simulator tracks each qubit's state independently, enabling efficient
    simulation of circuits without entangling gates.
    """

    __slots__ = ("n_qubits", "dtype", "qubit_states")

    def __init__(self, n_qubits: int, dtype=np.complex128):
        self.n_qubits = n_qubits
        self.dtype = dtype
        self.qubit_states = np.zeros((n_qubits, 2), dtype=dtype)
        self.qubit_states[:, 0] = 1.0

    def initialize(self, basis_state: int) -> None:
        self.qubit_states[:, :] = 0.0
        for i in range(self.n_qubits):
            bit = (basis_state >> (self.n_qubits - 1 - i)) & 1
            self.qubit_states[i, bit] = 1.0

    def initialize_from_product(self, single_qubit_states: list[np.ndarray]) -> None:
        if len(single_qubit_states) != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} states, got {len(single_qubit_states)}")
        for i, state in enumerate(single_qubit_states):
            if state.shape != (2,):
                raise ValueError(f"State {i} must have shape (2,), got {state.shape}")
            self.qubit_states[i] = state.astype(self.dtype)

    def apply_gate(self, gate_matrix: np.ndarray, qubit: int) -> None:
        if gate_matrix.shape != (2, 2):
            raise ValueError("Only single-qubit gates supported in product simulator")
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.n_qubits})")
        self.qubit_states[qubit] = gate_matrix @ self.qubit_states[qubit]

    def apply_operations(self, operations: list[GateOperation]) -> None:
        for op in operations:
            if len(op.targets) != 1:
                raise ValueError(
                    f"ProductStateSimulator only supports single-qubit gates, "
                    f"got {len(op.targets)}-qubit gate"
                )
            self.qubit_states[op.targets[0]] = op.matrix @ self.qubit_states[op.targets[0]]

    def apply_qft(self, start_qubit: int = 0, end_qubit: int | None = None) -> None:
        """
        Apply QFT directly in O(n) time by computing phases analytically.

        QFT|j⟩ produces a product state where qubit k has state:
            (|0⟩ + e^(2πij/2^(k+1))|1⟩) / √2

        This avoids O(n²) gate applications for QFT circuits.
        """
        if end_qubit is None:
            end_qubit = self.n_qubits
        n_qft = end_qubit - start_qubit
        j = 0
        for i in range(start_qubit, end_qubit):
            bit_val = int(np.abs(self.qubit_states[i, 1]) > 0.5)
            j = (j << 1) | bit_val
        sqrt2_inv = 1.0 / np.sqrt(2)
        for k in range(n_qft):
            qubit_idx = start_qubit + k
            phase = 2 * np.pi * j / (2 ** (k + 1))
            self.qubit_states[qubit_idx, 0] = sqrt2_inv
            self.qubit_states[qubit_idx, 1] = np.exp(1j * phase) * sqrt2_inv

    def apply_inverse_qft(self, start_qubit: int = 0, end_qubit: int | None = None) -> None:
        """Apply inverse QFT directly in O(n) time."""
        if end_qubit is None:
            end_qubit = self.n_qubits
        n_qft = end_qubit - start_qubit
        j = 0
        for i in range(start_qubit, end_qubit):
            bit_val = int(np.abs(self.qubit_states[i, 1]) > 0.5)
            j = (j << 1) | bit_val
        sqrt2_inv = 1.0 / np.sqrt(2)
        for k in range(n_qft):
            qubit_idx = start_qubit + k
            phase = -2 * np.pi * j / (2 ** (k + 1))
            self.qubit_states[qubit_idx, 0] = sqrt2_inv
            self.qubit_states[qubit_idx, 1] = np.exp(1j * phase) * sqrt2_inv

    def get_amplitude(self, basis_state: int) -> complex:
        if self.n_qubits >= _NUMBA_THRESHOLD:
            return _compute_amplitude_numba(self.qubit_states, basis_state, self.n_qubits)
        amp = self.dtype(1.0)
        for i in range(self.n_qubits):
            bit = (basis_state >> (self.n_qubits - 1 - i)) & 1
            amp *= self.qubit_states[i, bit]
        return amp

    def get_probabilities(self) -> np.ndarray:
        if self.n_qubits >= _NUMBA_THRESHOLD:
            return _compute_probabilities_parallel(self.qubit_states, self.n_qubits)
        size = 1 << self.n_qubits
        probs = np.ones(size, dtype=np.float64)
        for basis_state in range(size):
            for i in range(self.n_qubits):
                bit = (basis_state >> (self.n_qubits - 1 - i)) & 1
                probs[basis_state] *= np.abs(self.qubit_states[i, bit]) ** 2
        return probs

    def sample_array(self, shots: int) -> np.ndarray:
        qubit_probs_0 = np.abs(self.qubit_states[:, 0]) ** 2
        rng = np.random.default_rng()
        random_values = rng.random((shots, self.n_qubits))
        return _sample_product_state(qubit_probs_0, self.n_qubits, shots, random_values)

    def sample(self, shots: int) -> dict[str, int]:
        results = self.sample_array(shots)
        unique, counts = np.unique(results, return_counts=True)
        return {format(int(k), f"0{self.n_qubits}b"): int(v) for k, v in zip(unique, counts)}

    def get_qubit_state(self, qubit: int) -> np.ndarray:
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.n_qubits})")
        return self.qubit_states[qubit].copy()

    def get_bloch_vector(self, qubit: int) -> tuple[float, float, float]:
        state = self.get_qubit_state(qubit)
        alpha, beta = state[0], state[1]
        x = 2.0 * (alpha.conj() * beta).real
        y = 2.0 * (alpha.conj() * beta).imag
        z = (np.abs(alpha) ** 2 - np.abs(beta) ** 2).real
        return (float(x), float(y), float(z))

    def get_state_vector(self) -> np.ndarray:
        if self.n_qubits > 20:
            raise ValueError("State vector too large")
        if self.n_qubits >= _NUMBA_THRESHOLD:
            return _compute_state_vector_parallel(self.qubit_states, self.n_qubits)
        state = np.array([1.0], dtype=self.dtype)
        for i in range(self.n_qubits):
            state = np.kron(state, self.qubit_states[i])
        return state

    def copy(self) -> ProductStateSimulator:
        new_sim = ProductStateSimulator(self.n_qubits, self.dtype)
        new_sim.qubit_states = self.qubit_states.copy()
        return new_sim
