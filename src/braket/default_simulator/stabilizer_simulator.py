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
stabilizer_simulator.py

Efficient simulator for Clifford circuits using the stabilizer formalism.

This module implements the Gottesman-Knill theorem, enabling polynomial-time
simulation of circuits composed entirely of Clifford gates.

Mathematical Basis:
- Stabilizer tableau representation of quantum states
- Pauli group operations tracked via binary symplectic matrices
- Phase tracking for complete state representation

Complexity:
- Space: O(n^2) for n qubits
- Time: O(n^2) per Clifford gate, O(n^3) per measurement

Dependencies:
- numpy
- numba (for accelerated operations)
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation

_NUMBA_THRESHOLD = 16


@nb.njit(cache=True, fastmath=True)
def _rowsum_numba(
    tableau: np.ndarray, phases: np.ndarray, h: int, i: int, n: int
) -> None:  # pragma: no cover
    phase_sum = 0
    for j in range(n):
        x1 = tableau[i, j]
        z1 = tableau[i, n + j]
        x2 = tableau[h, j]
        z2 = tableau[h, n + j]

        if x1 == 0 and z1 == 0:
            g_val = 0
        elif x1 == 1 and z1 == 1:
            g_val = z2 - x2
        elif x1 == 1 and z1 == 0:
            g_val = z2 * (2 * x2 - 1)
        else:
            g_val = x2 * (1 - 2 * z2)
        phase_sum += g_val

    phases[h] = (phases[h] + phases[i] + phase_sum) % 4

    for j in range(2 * n):
        tableau[h, j] ^= tableau[i, j]


@nb.njit(cache=True)
def _hadamard_numba(
    tableau: np.ndarray, phases: np.ndarray, qubit: int, n: int
) -> None:  # pragma: no cover
    for i in range(2 * n):
        x = tableau[i, qubit]
        z = tableau[i, n + qubit]
        phases[i] = (phases[i] + 2 * x * z) % 4
        tableau[i, qubit] = z
        tableau[i, n + qubit] = x


@nb.njit(cache=True)
def _s_gate_numba(
    tableau: np.ndarray, phases: np.ndarray, qubit: int, n: int
) -> None:  # pragma: no cover
    for i in range(2 * n):
        x = tableau[i, qubit]
        z = tableau[i, n + qubit]
        phases[i] = (phases[i] + x * z) % 4
        tableau[i, n + qubit] = z ^ x


@nb.njit(cache=True)
def _cnot_numba(
    tableau: np.ndarray, phases: np.ndarray, control: int, target: int, n: int
) -> None:  # pragma: no cover
    for i in range(2 * n):
        x_c = tableau[i, control]
        z_c = tableau[i, n + control]
        x_t = tableau[i, target]
        z_t = tableau[i, n + target]
        phases[i] = (phases[i] + x_c * z_t * (x_t ^ z_c ^ 1)) % 4
        tableau[i, target] ^= x_c
        tableau[i, n + control] ^= z_t


class StabilizerTableau:
    """
    Stabilizer tableau representation of a quantum state.

    The tableau is a (2n) x (2n) binary matrix where:
    - Rows 0 to n-1 are destabilizers
    - Rows n to 2n-1 are stabilizers
    - Columns 0 to n-1 are X components
    - Columns n to 2n-1 are Z components

    Phases are tracked separately as integers mod 4.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.tableau = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=np.int8)
        self.phases = np.zeros(2 * n_qubits, dtype=np.int8)
        self._initialize_computational_basis()

    def _initialize_computational_basis(self) -> None:
        n = self.n
        for i in range(n):
            self.tableau[i, i] = 1
            self.tableau[n + i, n + i] = 1
        self.phases[:] = 0

    def _rowsum(self, h: int, i: int) -> None:
        if self.n >= _NUMBA_THRESHOLD:
            _rowsum_numba(self.tableau, self.phases, h, i, self.n)
            return

        n = self.n

        def g(x1, z1, x2, z2):
            if x1 == 0 and z1 == 0:
                return 0
            elif x1 == 1 and z1 == 1:
                return z2 - x2
            elif x1 == 1 and z1 == 0:
                return z2 * (2 * x2 - 1)
            else:
                return x2 * (1 - 2 * z2)

        phase_sum = 0
        for j in range(n):
            x1 = self.tableau[i, j]
            z1 = self.tableau[i, n + j]
            x2 = self.tableau[h, j]
            z2 = self.tableau[h, n + j]
            phase_sum += g(x1, z1, x2, z2)

        new_phase = (self.phases[h] + self.phases[i] + phase_sum) % 4
        self.phases[h] = new_phase

        for j in range(2 * n):
            self.tableau[h, j] ^= self.tableau[i, j]

    def h(self, qubit: int) -> None:
        if self.n >= _NUMBA_THRESHOLD:
            _hadamard_numba(self.tableau, self.phases, qubit, self.n)
            return

        n = self.n
        for i in range(2 * n):
            x = self.tableau[i, qubit]
            z = self.tableau[i, n + qubit]
            self.phases[i] = (self.phases[i] + 2 * x * z) % 4
            self.tableau[i, qubit] = z
            self.tableau[i, n + qubit] = x

    def s(self, qubit: int) -> None:
        if self.n >= _NUMBA_THRESHOLD:
            _s_gate_numba(self.tableau, self.phases, qubit, self.n)
            return

        n = self.n
        for i in range(2 * n):
            x = self.tableau[i, qubit]
            z = self.tableau[i, n + qubit]
            self.phases[i] = (self.phases[i] + x * z) % 4
            self.tableau[i, n + qubit] = z ^ x

    def sdg(self, qubit: int) -> None:
        self.s(qubit)
        self.s(qubit)
        self.s(qubit)

    def x(self, qubit: int) -> None:
        n = self.n
        for i in range(2 * n):
            z = self.tableau[i, n + qubit]
            self.phases[i] = (self.phases[i] + 2 * z) % 4

    def y(self, qubit: int) -> None:
        n = self.n
        for i in range(2 * n):
            x = self.tableau[i, qubit]
            z = self.tableau[i, n + qubit]
            self.phases[i] = (self.phases[i] + 2 * (x ^ z)) % 4

    def z(self, qubit: int) -> None:
        n = self.n
        for i in range(2 * n):
            x = self.tableau[i, qubit]
            self.phases[i] = (self.phases[i] + 2 * x) % 4

    def cnot(self, control: int, target: int) -> None:
        if self.n >= _NUMBA_THRESHOLD:
            _cnot_numba(self.tableau, self.phases, control, target, self.n)
            return

        n = self.n
        for i in range(2 * n):
            xc = self.tableau[i, control]
            zc = self.tableau[i, n + control]
            xt = self.tableau[i, target]
            zt = self.tableau[i, n + target]

            self.phases[i] = (self.phases[i] + xc * zt * (xt ^ zc ^ 1)) % 4
            self.tableau[i, target] = xt ^ xc
            self.tableau[i, n + control] = zc ^ zt

    def cz(self, qubit1: int, qubit2: int) -> None:
        self.h(qubit2)
        self.cnot(qubit1, qubit2)
        self.h(qubit2)

    def swap(self, qubit1: int, qubit2: int) -> None:
        self.cnot(qubit1, qubit2)
        self.cnot(qubit2, qubit1)
        self.cnot(qubit1, qubit2)

    def measure(self, qubit: int, random_outcome: int | None = None) -> int:
        n = self.n

        p = None
        for i in range(n, 2 * n):
            if self.tableau[i, qubit] == 1:
                p = i
                break

        if p is not None:
            for i in range(2 * n):
                if i != p and self.tableau[i, qubit] == 1:
                    self._rowsum(i, p)

            self.tableau[p - n] = self.tableau[p].copy()
            self.phases[p - n] = self.phases[p]

            self.tableau[p] = 0
            self.phases[p] = 0
            self.tableau[p, n + qubit] = 1

            if random_outcome is None:
                random_outcome = np.random.randint(0, 2)

            self.phases[p] = 2 * random_outcome
            return random_outcome

        else:
            scratch = np.zeros(2 * n, dtype=np.int8)
            scratch_phase = 0

            for i in range(n):
                if self.tableau[i, qubit] == 1:
                    if scratch_phase == 0 and np.sum(scratch) == 0:
                        scratch = self.tableau[n + i].copy()
                        scratch_phase = self.phases[n + i]
                    else:
                        for j in range(2 * n):
                            scratch[j] ^= self.tableau[n + i, j]

            return (scratch_phase // 2) % 2

    def copy(self) -> StabilizerTableau:
        new_tableau = StabilizerTableau(self.n)
        new_tableau.tableau = self.tableau.copy()
        new_tableau.phases = self.phases.copy()
        return new_tableau


class StabilizerSimulator:
    """
    Simulator for Clifford circuits using stabilizer formalism.
    """

    GATE_MAP = {
        "hadamard": "h",
        "h": "h",
        "pauli_x": "x",
        "x": "x",
        "pauli_y": "y",
        "y": "y",
        "pauli_z": "z",
        "z": "z",
        "s": "s",
        "si": "sdg",
        "sdg": "sdg",
        "cx": "cnot",
        "cnot": "cnot",
        "cz": "cz",
        "swap": "swap",
    }

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.tableau = StabilizerTableau(n_qubits)

    def initialize(self, basis_state: int) -> None:
        self.tableau = StabilizerTableau(self.n_qubits)
        for i in range(self.n_qubits):
            bit = (basis_state >> (self.n_qubits - 1 - i)) & 1
            if bit == 1:
                self.tableau.x(i)

    def apply_gate(self, gate_name: str, qubits: list[int]) -> None:
        gate_name_lower = gate_name.lower()
        mapped_gate = self.GATE_MAP.get(gate_name_lower, gate_name_lower)

        if mapped_gate == "h":
            self.tableau.h(qubits[0])
        elif mapped_gate == "x":
            self.tableau.x(qubits[0])
        elif mapped_gate == "y":
            self.tableau.y(qubits[0])
        elif mapped_gate == "z":
            self.tableau.z(qubits[0])
        elif mapped_gate == "s":
            self.tableau.s(qubits[0])
        elif mapped_gate == "sdg":
            self.tableau.sdg(qubits[0])
        elif mapped_gate == "cnot":
            self.tableau.cnot(qubits[0], qubits[1])
        elif mapped_gate == "cz":
            self.tableau.cz(qubits[0], qubits[1])
        elif mapped_gate == "swap":
            self.tableau.swap(qubits[0], qubits[1])
        else:
            raise ValueError(f"Unsupported Clifford gate: {gate_name}")

    def apply_operations(self, operations: list[GateOperation]) -> None:
        for op in operations:
            gate_type = getattr(op, "gate_type", "unknown")
            self.apply_gate(gate_type, list(op.targets))

    def measure(self, qubit: int) -> int:
        return self.tableau.measure(qubit)

    def measure_all(self) -> str:
        result = ""
        tableau_copy = self.tableau.copy()
        for i in range(self.n_qubits):
            bit = tableau_copy.measure(i)
            result += str(bit)
        return result

    def sample(self, shots: int) -> dict[str, int]:
        results = []
        for _ in range(shots):
            tableau_copy = self.tableau.copy()
            outcome = ""
            for i in range(self.n_qubits):
                bit = tableau_copy.measure(i)
                outcome += str(bit)
            results.append(outcome)

        return dict(Counter(results))

    def is_deterministic(self, qubit: int) -> tuple[bool, int | None]:
        n = self.n_qubits
        for i in range(n, 2 * n):
            if self.tableau.tableau[i, qubit] == 1:
                return (False, None)

        scratch = np.zeros(2 * n, dtype=np.int8)
        scratch_phase = 0

        for i in range(n):
            if self.tableau.tableau[i, qubit] == 1:
                if scratch_phase == 0 and np.sum(scratch) == 0:
                    scratch = self.tableau.tableau[n + i].copy()
                    scratch_phase = self.tableau.phases[n + i]
                else:
                    for j in range(2 * n):
                        scratch[j] ^= self.tableau.tableau[n + i, j]

        return (True, (scratch_phase // 2) % 2)

    def copy(self) -> StabilizerSimulator:
        new_sim = StabilizerSimulator(self.n_qubits)
        new_sim.tableau = self.tableau.copy()
        return new_sim

    def get_state_vector(self) -> np.ndarray:
        if self.n_qubits > 20:
            raise ValueError("State vector too large to compute explicitly")

        dim = 2**self.n_qubits
        state = np.zeros(dim, dtype=np.complex128)

        for basis_idx in range(dim):
            tableau_copy = self.tableau.copy()
            amplitude = 1.0 + 0j
            valid = True

            for q in range(self.n_qubits):
                bit = (basis_idx >> (self.n_qubits - 1 - q)) & 1
                is_det, det_val = self._is_deterministic_copy(tableau_copy, q)

                if is_det:
                    if det_val != bit:
                        valid = False
                        break
                    tableau_copy.measure(q, random_outcome=bit)
                else:
                    amplitude *= 1.0 / np.sqrt(2)
                    tableau_copy.measure(q, random_outcome=bit)

            if valid:
                state[basis_idx] = amplitude

        state /= np.linalg.norm(state)
        return state

    def _is_deterministic_copy(
        self, tableau: StabilizerTableau, qubit: int
    ) -> tuple[bool, int | None]:
        n = tableau.n
        for i in range(n, 2 * n):
            if tableau.tableau[i, qubit] == 1:
                return (False, None)

        scratch = np.zeros(2 * n, dtype=np.int8)
        scratch_phase = 0

        for i in range(n):
            if tableau.tableau[i, qubit] == 1:
                if scratch_phase == 0 and np.sum(scratch) == 0:
                    scratch = tableau.tableau[n + i].copy()
                    scratch_phase = tableau.phases[n + i]
                else:
                    for j in range(2 * n):
                        scratch[j] ^= tableau.tableau[n + i, j]

        return (True, (scratch_phase // 2) % 2)
