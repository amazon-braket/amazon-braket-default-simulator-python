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

from __future__ import annotations

import cmath
import math
from functools import singledispatch
from typing import Tuple

import numpy as np

import braket.ir.jaqcd as braket_instruction
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.operation_helpers import (
    check_matrix_dimensions,
    check_unitary,
    ir_matrix_to_ndarray,
)


def from_braket_instruction(instruction) -> GateOperation:
    """Instantiates the concrete `GateOperation` object from the specified braket instruction.

    Args:
        instruction: instruction for a circuit specified using the `braket.ir.jacqd` format.
    Returns:
        GateOperation: instance of the concrete GateOperation class corresponding to
        the specified instruction.

    Raises:
        ValueError: If no concrete `GateOperation` class has been registered
            for the instruction type.
    """
    return _from_braket_instruction(instruction)


@singledispatch
def _from_braket_instruction(instruction):
    raise ValueError(f"Instruction {instruction} not recognized")


class Identity(GateOperation):
    """Identity gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.eye(2)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.I)
def _i(instruction) -> Identity:
    return Identity([instruction.target])


class Hadamard(GateOperation):
    """Hadamard gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]]) / math.sqrt(2)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.H)
def _hadamard(instruction) -> Hadamard:
    return Hadamard([instruction.target])


class PauliX(GateOperation):
    """Pauli-X gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.X)
def _pauli_x(instruction) -> PauliX:
    return PauliX([instruction.target])


class PauliY(GateOperation):
    """Pauli-Y gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Y)
def _pauli_y(instruction) -> PauliY:
    return PauliY([instruction.target])


class PauliZ(GateOperation):
    """Pauli-Z gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Z)
def _pauli_z(instruction) -> PauliZ:
    return PauliZ([instruction.target])


class CX(GateOperation):
    """Controlled Pauli-X gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CNot)
def _cx(instruction) -> CX:
    return CX([instruction.control, instruction.target])


class CY(GateOperation):
    """Controlled Pauli-Y gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CY)
def _cy(instruction) -> CY:
    return CY([instruction.control, instruction.target])


class CZ(GateOperation):
    """Controlled Pauli-Z gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CZ)
def _cz(instruction) -> CZ:
    return CZ([instruction.control, instruction.target])


class S(GateOperation):
    """S gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=complex)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.S)
def _s(instruction) -> S:
    return S([instruction.target])


class Si(GateOperation):
    r"""The adjoint :math:`S^{\dagger}` of the S gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1j]], dtype=complex)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Si)
def _si(instruction) -> Si:
    return Si([instruction.target])


class T(GateOperation):
    """T gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.T)
def _t(instruction) -> T:
    return T([instruction.target])


class Ti(GateOperation):
    r"""The adjoint :math:`T^{\dagger}` of the T gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=complex)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Ti)
def _ti(instruction) -> Ti:
    return Ti([instruction.target])


class V(GateOperation):
    """Square root of the X (not) gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.V)
def _v(instruction) -> V:
    return V([instruction.target])


class Vi(GateOperation):
    r"""The adjoint :math:`V^{\dagger}` of the square root of the X (not) gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]), dtype=complex)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Vi)
def _vi(instruction) -> Vi:
    return Vi([instruction.target])


class PhaseShift(GateOperation):
    """Phase shift gate"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, cmath.exp(1j * self._angle)]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.PhaseShift)
def _phase_shift(instruction) -> PhaseShift:
    return PhaseShift([instruction.target], instruction.angle)


class CPhaseShift(GateOperation):
    """Controlled phase shift gate"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([1.0, 1.0, 1.0, cmath.exp(1j * self._angle)])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CPhaseShift)
def _c_phase_shift(instruction) -> CPhaseShift:
    return CPhaseShift([instruction.control, instruction.target], instruction.angle)


class CPhaseShift00(GateOperation):
    r"""Controlled phase shift gate phasing the phasing :math:`\ket{00}` state"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([cmath.exp(1j * self._angle), 1.0, 1.0, 1.0])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CPhaseShift00)
def _c_phase_shift_00(instruction) -> CPhaseShift00:
    return CPhaseShift00([instruction.control, instruction.target], instruction.angle)


class CPhaseShift01(GateOperation):
    r"""Controlled phase shift gate phasing the phasing :math:`\ket{01}` state"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([1.0, cmath.exp(1j * self._angle), 1.0, 1.0])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CPhaseShift01)
def _c_phase_shift_01(instruction) -> CPhaseShift01:
    return CPhaseShift01([instruction.control, instruction.target], instruction.angle)


class CPhaseShift10(GateOperation):
    r"""Controlled phase shift gate phasing the phasing :math:`\ket{10}` state"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([1.0, 1.0, cmath.exp(1j * self._angle), 1.0])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CPhaseShift10)
def _c_phase_shift_10(instruction) -> CPhaseShift10:
    return CPhaseShift10([instruction.control, instruction.target], instruction.angle)


class RotX(GateOperation):
    """X-axis rotation gate"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_half_angle = math.cos(self._angle / 2)
        i_sin_half_angle = 1j * math.sin(self._angle / 2)
        return np.array([[cos_half_angle, -i_sin_half_angle], [-i_sin_half_angle, cos_half_angle]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Rx)
def _rot_x(instruction) -> RotX:
    return RotX([instruction.target], instruction.angle)


class RotY(GateOperation):
    """Y-axis rotation gate"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_half_angle = math.cos(self._angle / 2)
        sin_half_angle = math.sin(self._angle / 2)
        return np.array([[cos_half_angle, -sin_half_angle], [sin_half_angle, cos_half_angle]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Ry)
def _rot_y(instruction) -> RotY:
    return RotY([instruction.target], instruction.angle)


class RotZ(GateOperation):
    """Z-axis rotation gate"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        positive_phase = cmath.exp(1j * self._angle / 2)
        negative_phase = cmath.exp(-1j * self._angle / 2)
        return np.array([[negative_phase, 0], [0, positive_phase]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Rz)
def _rot_z(instruction) -> RotZ:
    return RotZ([instruction.target], instruction.angle)


class Swap(GateOperation):
    """Swap gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Swap)
def _swap(instruction) -> Swap:
    return Swap(instruction.targets)


class ISwap(GateOperation):
    """ISwap gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0j, 0.0],
                [0.0, 1.0j, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.ISwap)
def _iswap(instruction) -> ISwap:
    return ISwap(instruction.targets)


class PSwap(GateOperation):
    """Parametrized Swap gate"""

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, cmath.exp(1j * self._angle), 0.0],
                [0.0, cmath.exp(1j * self._angle), 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.PSwap)
def _pswap(instruction) -> PSwap:
    return PSwap(instruction.targets, instruction.angle)


class XY(GateOperation):
    """XY gate

    Reference: https://arxiv.org/abs/1912.04424v1
    """

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos = math.cos(self._angle / 2)
        sin = math.sin(self._angle / 2)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, 1.0j * sin, 0.0],
                [0.0, 1.0j * sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.XY)
def _xy(instruction) -> XY:
    return XY(instruction.targets, instruction.angle)


class XX(GateOperation):
    """Ising XX gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_angle = math.cos(self._angle / 2)
        i_sin_angle = 1j * math.sin(self._angle / 2)
        return np.array(
            [
                [cos_angle, 0, 0, -i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [-i_sin_angle, 0, 0, cos_angle],
            ]
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.XX)
def _xx(instruction) -> XX:
    return XX(instruction.targets, instruction.angle)


class YY(GateOperation):
    """Ising YY gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_angle = math.cos(self._angle / 2)
        i_sin_angle = 1j * math.sin(self._angle / 2)
        return np.array(
            [
                [cos_angle, 0, 0, i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [i_sin_angle, 0, 0, cos_angle],
            ]
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.YY)
def _yy(instruction) -> YY:
    return YY(instruction.targets, instruction.angle)


class ZZ(GateOperation):
    """Ising ZZ gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle):
        self._targets = tuple(targets)
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        positive_phase = cmath.exp(1j * self._angle / 2)
        negative_phase = cmath.exp(-1j * self._angle / 2)
        return np.array(
            [
                [negative_phase, 0, 0, 0],
                [0, positive_phase, 0, 0],
                [0, 0, positive_phase, 0],
                [0, 0, 0, negative_phase],
            ]
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.ZZ)
def _zz(instruction) -> ZZ:
    return ZZ(instruction.targets, instruction.angle)


class CCNot(GateOperation):
    """Controlled CNOT or Toffoli gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=complex,
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CCNot)
def _ccnot(instruction) -> CCNot:
    return CCNot([*instruction.controls, instruction.target])


class CSwap(GateOperation):
    """Controlled Swap gate"""

    def __init__(self, targets):
        self._targets = tuple(targets)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=complex,
        )

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.CSwap)
def _cswap(instruction) -> CSwap:
    return CSwap([instruction.control, *instruction.targets])


class Unitary(GateOperation):
    """Unitary gate"""

    def __init__(self, targets, matrix):
        self._targets = tuple(targets)
        clone = np.array(matrix, dtype=complex)
        check_matrix_dimensions(clone, self._targets)
        check_unitary(clone)
        self._matrix = clone

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self._matrix)

    @property
    def targets(self) -> Tuple[int, ...]:
        return self._targets


@_from_braket_instruction.register(braket_instruction.Unitary)
def _unitary(instruction) -> Unitary:
    return Unitary(instruction.targets, ir_matrix_to_ndarray(instruction.matrix))
