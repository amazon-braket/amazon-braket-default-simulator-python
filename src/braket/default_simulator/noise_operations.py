from __future__ import annotations

import cmath
from typing import Tuple

import braket.ir.jaqcd as braket_instruction
import numpy as np
from braket.default_simulator.operation import KrausOperation
from braket.default_simulator.operation_helpers import (
    check_matrix_dimensions,
    check_CPTP,
    ir_matrix_to_ndarray,
    _from_braket_instruction,
)


class Bit_Flip(KrausOperation):
    """Bit Flip noise channel"""

    def __init__(self, targets, prob):
        self._targets = tuple(targets)
        self._prob = prob

    @property
    def matrices(self) -> np.ndarray:
        k0 = np.sqrt(1-self._prob) * np.array([[1, 0], [0, 1]])
        k1 = np.sqrt(self._prob)   * np.array([[0, 1], [1, 0]])
        return [k0, k1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets

@_from_braket_instruction.register(braket_instruction.Bit_Flip)
def _bit_flip(instruction) -> Bit_Flip:
    return Bit_Flip([instruction.target], instruction.prob)


class Phase_Flip(KrausOperation):
    """Phase Flip noise channel"""

    def __init__(self, targets, prob):
        self._targets = tuple(targets)
        self._prob = prob

    @property
    def matrices(self) -> np.ndarray:
        k0 = np.sqrt(1-self._prob) * np.array([[1.0, 0.0], [0.0, 1.0]])
        k1 = np.sqrt(self._prob)   * np.array([[1.0, 0.0], [0.0, -1.0]])
        return [k0, k1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets

@_from_braket_instruction.register(braket_instruction.Phase_Flip)
def _phase_flip(instruction) -> Phase_Flip:
    return Phase_Flip([instruction.target], instruction.prob)


class Depolarizing(KrausOperation):
    """Depolarizing noise channel"""

    def __init__(self, targets, prob):
        self._targets = tuple(targets)
        self._prob = prob

    @property
    def matrices(self) -> np.ndarray:
        K0 = np.sqrt(1-self._prob)      * np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        K1 = np.sqrt(self._prob/3)      * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        K2 = np.sqrt(self._prob/3) * 1j * np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
        K3 = np.sqrt(self._prob/3)      * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        return [K0, K1, K2, K3]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets

@_from_braket_instruction.register(braket_instruction.Depolarizing)
def _depolarizing(instruction) -> Depolarizing:
    return Depolarizing([instruction.target], instruction.prob)


class Amplitude_Damping(KrausOperation):
    """Amplitude Damping noise channel"""

    def __init__(self, targets, prob):
        self._targets = tuple(targets)
        self._prob = prob

    @property
    def matrices(self) -> np.ndarray:
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1-self._prob)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(self._prob)], [0.0, 0.0]], dtype=complex)
        return [K0, K1]

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets

@_from_braket_instruction.register(braket_instruction.Amplitude_Damping)
def _amplitude_damping(instruction) -> Amplitude_Damping:
    return Amplitude_Damping([instruction.target], instruction.prob)


class Kraus(KrausOperation):
    """Arbitrary quantum channel"""

    def __init__(self, targets, matrices):
        self._targets = tuple(targets)
        clone = [np.array(matrix, dtype=complex) for matrix in matrices]
        for matrix in clone:
            check_matrix_dimensions(matrix, self._targets)
        check_CPTP(clone)
        self._matrices = clone

    @property
    def matrices(self) -> np.ndarray:
        return self._matrices

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets

@_from_braket_instruction.register(braket_instruction.Kraus)
def _kraus(instruction) -> Kraus:
    return Kraus(instruction.targets, [ir_matrix_to_ndarray(matrix) for matrix in instruction.matrices])
