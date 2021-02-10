from __future__ import annotations

from typing import Tuple

import numpy as np

import braket.ir.jaqcd as braket_instruction
from braket.default_simulator.operation import KrausOperation
from braket.default_simulator.operation_helpers import (
    _from_braket_instruction,
    check_cptp,
    check_matrix_dimensions,
    ir_matrix_to_ndarray,
)


class BitFlip(KrausOperation):
    """Bit Flip noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> np.ndarray:
        k0 = np.sqrt(1 - self._probability) * np.array([[1, 0], [0, 1]])
        k1 = np.sqrt(self._probability) * np.array([[0, 1], [1, 0]])
        return [k0, k1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.BitFlip)
def _bit_flip(instruction) -> BitFlip:
    return BitFlip([instruction.target], instruction.probability)


class PhaseFlip(KrausOperation):
    """Phase Flip noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> np.ndarray:
        k0 = np.sqrt(1 - self._probability) * np.array([[1.0, 0.0], [0.0, 1.0]])
        k1 = np.sqrt(self._probability) * np.array([[1.0, 0.0], [0.0, -1.0]])
        return [k0, k1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.PhaseFlip)
def _phase_flip(instruction) -> PhaseFlip:
    return PhaseFlip([instruction.target], instruction.probability)


class Depolarizing(KrausOperation):
    """Depolarizing noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> np.ndarray:
        K0 = np.sqrt(1 - self._probability) * np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        K1 = np.sqrt(self._probability / 3) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        K2 = (
            np.sqrt(self._probability / 3) * 1j * np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
        )
        K3 = np.sqrt(self._probability / 3) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        return [K0, K1, K2, K3]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Depolarizing)
def _depolarizing(instruction) -> Depolarizing:
    return Depolarizing([instruction.target], instruction.probability)


class AmplitudeDamping(KrausOperation):
    """Amplitude Damping noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> np.ndarray:
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - self._probability)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(self._probability)], [0.0, 0.0]], dtype=complex)
        return [K0, K1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.AmplitudeDamping)
def _amplitude_damping(instruction) -> AmplitudeDamping:
    return AmplitudeDamping([instruction.target], instruction.probability)


class PhaseDamping(KrausOperation):
    """Phase Damping noise channel"""

    def __init__(self, targets, probability):
        self._targets = tuple(targets)
        self._probability = probability

    @property
    def matrices(self) -> np.ndarray:
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - self._probability)]], dtype=complex)
        K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(self._probability)]], dtype=complex)
        return [K0, K1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.PhaseDamping)
def _phase_damping(instruction) -> PhaseDamping:
    return PhaseDamping([instruction.target], instruction.probability)


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
    def matrices(self) -> np.ndarray:
        return self._matrices

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Kraus)
def _kraus(instruction) -> Kraus:
    return Kraus(
        instruction.targets, [ir_matrix_to_ndarray(matrix) for matrix in instruction.matrices]
    )
