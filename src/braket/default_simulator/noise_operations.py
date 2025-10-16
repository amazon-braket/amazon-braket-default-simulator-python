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

import itertools

import numpy as np

import braket.ir.jaqcd as braket_instruction
from braket.default_simulator.operation import KrausOperation
from braket.default_simulator.operation_helpers import (
    _from_braket_instruction,
    check_cptp,
    check_matrix_dimensions,
    ir_matrix_to_ndarray,
)

_PAULI_I = np.eye(2)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_PAULI_Z = np.diag([1.0, -1.0])


class BitFlip(KrausOperation):
    """Bit Flip noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> list[np.ndarray]:
        return [np.sqrt(1 - self._probability) * _PAULI_I, np.sqrt(self._probability) * _PAULI_X]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probability(self):
        return self._probability


@_from_braket_instruction.register(braket_instruction.BitFlip)
def _bit_flip(instruction) -> BitFlip:
    return BitFlip([instruction.target], instruction.probability)


class PhaseFlip(KrausOperation):
    """Phase Flip noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> list[np.ndarray]:
        return [np.sqrt(1 - self._probability) * _PAULI_I, np.sqrt(self._probability) * _PAULI_Z]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probability(self):
        return self._probability


@_from_braket_instruction.register(braket_instruction.PhaseFlip)
def _phase_flip(instruction) -> PhaseFlip:
    return PhaseFlip([instruction.target], instruction.probability)


class PauliChannel(KrausOperation):
    """Pauli noise channel"""

    def __init__(self, targets, probX, probY, probZ):
        self._targets = tuple(targets)
        self._probX = probX
        self._probY = probY
        self._probZ = probZ

    @property
    def matrices(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self._probX - self._probY - self._probZ) * _PAULI_I,
            np.sqrt(self._probX) * _PAULI_X,
            np.sqrt(self._probY) * _PAULI_Y,
            np.sqrt(self._probZ) * _PAULI_Z,
        ]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probabilities(self):
        return [self._probX, self._probY, self._probZ]


@_from_braket_instruction.register(braket_instruction.PauliChannel)
def _pauli_channel(instruction) -> PauliChannel:
    return PauliChannel(
        [instruction.target], instruction.probX, instruction.probY, instruction.probZ
    )


class Depolarizing(KrausOperation):
    """Depolarizing noise channel"""

    _k2_base = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self._probability) * _PAULI_I,
            np.sqrt(self._probability / 3) * _PAULI_X,
            np.sqrt(self._probability / 3) * 1j * Depolarizing._k2_base,
            np.sqrt(self._probability / 3) * _PAULI_Z,
        ]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probability(self):
        return self._probability


@_from_braket_instruction.register(braket_instruction.Depolarizing)
def _depolarizing(instruction) -> Depolarizing:
    return Depolarizing([instruction.target], instruction.probability)


class TwoQubitDepolarizing(KrausOperation):
    """Two-qubit Depolarizing noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> list[np.ndarray]:
        k_list_single = [_PAULI_I, _PAULI_X, _PAULI_Y, _PAULI_Z]
        k_list = [np.kron(i, j) for i in k_list_single for j in k_list_single]
        k_list[0] *= np.sqrt(1 - self._probability)
        k_list[1:] = [np.sqrt(self._probability / 15) * i for i in k_list[1:]]
        return k_list

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probability(self):
        return self._probability


@_from_braket_instruction.register(braket_instruction.TwoQubitDepolarizing)
def _two_qubit_depolarizing(instruction) -> TwoQubitDepolarizing:
    return TwoQubitDepolarizing(instruction.targets, instruction.probability)


class TwoQubitDephasing(KrausOperation):
    """Two-qubit Dephasing noise channel"""

    _k0_base = np.eye(4)
    _k1_base = np.kron(_PAULI_I, _PAULI_Z)
    _k2_base = np.kron(_PAULI_Z, _PAULI_I)
    _k3_base = np.kron(_PAULI_Z, _PAULI_Z)

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self._probability) * TwoQubitDephasing._k0_base,
            np.sqrt(self._probability / 3) * TwoQubitDephasing._k1_base,
            np.sqrt(self._probability / 3) * TwoQubitDephasing._k2_base,
            np.sqrt(self._probability / 3) * TwoQubitDephasing._k3_base,
        ]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probability(self):
        return self._probability


@_from_braket_instruction.register(braket_instruction.TwoQubitDephasing)
def _two_qubit_dephasing(instruction) -> TwoQubitDephasing:
    return TwoQubitDephasing(instruction.targets, instruction.probability)


class AmplitudeDamping(KrausOperation):
    """Amplitude Damping noise channel"""

    def __init__(self, targets, gamma):
        self._targets = tuple(targets)
        self._gamma = gamma

    @property
    def matrices(self) -> list[np.ndarray]:
        return [
            np.array([[1.0, 0.0], [0.0, np.sqrt(1 - self._gamma)]], dtype=complex),
            np.array([[0.0, np.sqrt(self._gamma)], [0.0, 0.0]], dtype=complex),
        ]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def gamma(self):
        return self._gamma


@_from_braket_instruction.register(braket_instruction.AmplitudeDamping)
def _amplitude_damping(instruction) -> AmplitudeDamping:
    return AmplitudeDamping([instruction.target], instruction.gamma)


class GeneralizedAmplitudeDamping(KrausOperation):
    """Generalized Amplitude Damping noise channel"""

    def __init__(self, targets, gamma, probability):
        self._targets = tuple(targets)
        self._gamma = gamma
        self._probability = probability

    @property
    def matrices(self) -> list[np.ndarray]:
        return [
            np.sqrt(self._probability)
            * np.array([[1.0, 0.0], [0.0, np.sqrt(1 - self._gamma)]], dtype=complex),
            np.sqrt(self._probability)
            * np.array([[0.0, np.sqrt(self._gamma)], [0.0, 0.0]], dtype=complex),
            np.sqrt(1 - self._probability)
            * np.array([[np.sqrt(1 - self._gamma), 0.0], [0.0, 1.0]], dtype=complex),
            np.sqrt(1 - self._probability)
            * np.array([[0.0, 0.0], [np.sqrt(self._gamma), 0.0]], dtype=complex),
        ]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def probability(self):
        return self._probability

    @property
    def gamma(self):
        return self._gamma


@_from_braket_instruction.register(braket_instruction.GeneralizedAmplitudeDamping)
def _generalized_amplitude_damping(instruction) -> GeneralizedAmplitudeDamping:
    return GeneralizedAmplitudeDamping(
        [instruction.target], instruction.gamma, instruction.probability
    )


class PhaseDamping(KrausOperation):
    """Phase Damping noise channel"""

    def __init__(self, targets, gamma):
        self._targets = tuple(targets)
        self._gamma = gamma

    @property
    def matrices(self) -> list[np.ndarray]:
        return [
            np.array([[1.0, 0.0], [0.0, np.sqrt(1 - self._gamma)]], dtype=complex),
            np.array([[0.0, 0.0], [0.0, np.sqrt(self._gamma)]], dtype=complex),
        ]

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets

    @property
    def gamma(self):
        return self._gamma


@_from_braket_instruction.register(braket_instruction.PhaseDamping)
def _phase_damping(instruction) -> PhaseDamping:
    return PhaseDamping([instruction.target], instruction.gamma)


class Kraus(KrausOperation):
    """Arbitrary quantum channel that evolve a density matrix through the operator-sum
    formalism with the provided matrices as Kraus operators.
    """

    def __init__(self, targets, matrices):
        self._targets = tuple(targets)
        clone = [np.array(matrix, dtype=complex) for matrix in matrices]
        for matrix in clone:
            check_matrix_dimensions(matrix, self._targets)
        check_cptp(clone)
        self._matrices = clone

    @property
    def matrices(self) -> list[np.ndarray]:
        return self._matrices

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Kraus)
def _kraus(instruction) -> Kraus:
    return Kraus(
        instruction.targets, [ir_matrix_to_ndarray(matrix) for matrix in instruction.matrices]
    )


class TwoQubitPauliChannel(KrausOperation):
    """Two qubit Pauli noise channel"""

    _paulis = {
        "I": _PAULI_I,
        "X": _PAULI_X,
        "Y": _PAULI_Y,
        "Z": _PAULI_Z,
    }
    _tensor_products_strings = itertools.product(_paulis.keys(), repeat=2)
    _names_list = ["".join(x) for x in _tensor_products_strings]

    def __init__(self, targets, probabilities):
        self._targets = tuple(targets)
        self.probabilities = probabilities

        total_prob = sum(self.probabilities.values())

        k_list = [np.sqrt(1 - total_prob) * np.identity(4)]  # identity
        for pstring in self._names_list[1:]:  # ignore "II"
            if pstring in self.probabilities:
                mat = np.sqrt(self.probabilities[pstring]) * np.kron(
                    self._paulis[pstring[0]], self._paulis[pstring[1]]
                )
                k_list.append(mat)
            else:
                k_list.append(np.zeros((4, 4)))
        self._matrices = k_list

    @property
    def matrices(self) -> list[np.ndarray]:
        return self._matrices

    @property
    def targets(self) -> tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.MultiQubitPauliChannel)
def _two_qubit_pauli_channel(instruction) -> TwoQubitPauliChannel:
    return TwoQubitPauliChannel(instruction.targets, instruction.probabilities)
