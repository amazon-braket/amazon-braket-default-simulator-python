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

from functools import singledispatch
from typing import List

import braket.ir.jaqcd as braket_instruction
import numpy as np
from braket.default_simulator.gate_operation import GateOperation


@singledispatch
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
    raise ValueError(f"Instruction {instruction} not recognized")


class Hadamard(GateOperation):
    """Hadamard gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.H)
def _hadamard(instruction) -> Hadamard:
    return Hadamard([instruction.target])


class PauliX(GateOperation):
    """Pauli-X gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.X)
def _pauli_x(instruction) -> PauliX:
    return PauliX([instruction.target])


class PauliY(GateOperation):
    """Pauli-Y gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Y)
def _pauli_y(instruction) -> PauliY:
    return PauliY([instruction.target])


class PauliZ(GateOperation):
    """Pauli-Z gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Z)
def _pauli_z(instruction) -> PauliZ:
    return PauliZ([instruction.target])


class CX(GateOperation):
    """Controlled Pauli-X gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    @property
    def targets(self) -> List[int]:
        return self._targets

    @property
    def control(self) -> int:
        return self._targets[0]


@from_braket_instruction.register(braket_instruction.CNot)
def _cx(instruction) -> CX:
    return CX([instruction.control, instruction.target])


class CY(GateOperation):
    """Controlled Pauli-Y gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])

    @property
    def targets(self) -> List[int]:
        return self._targets

    @property
    def control(self) -> int:
        return self._targets[0]


@from_braket_instruction.register(braket_instruction.CY)
def _cy(instruction) -> CY:
    return CY([instruction.control, instruction.target])


class CZ(GateOperation):
    """Controlled Pauli-Z gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    @property
    def targets(self) -> List[int]:
        return self._targets

    @property
    def control(self) -> int:
        return self._targets[0]


@from_braket_instruction.register(braket_instruction.CZ)
def _cz(instruction) -> CZ:
    return CZ([instruction.control, instruction.target])


class T(GateOperation):
    """T gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.T)
def _t(instruction) -> T:
    return T([instruction.target])


class S(GateOperation):
    """S gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.S)
def _s(instruction) -> S:
    return S([instruction.target])


class PhaseShift(GateOperation):
    """Phase shift gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * self._angle)]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.PhaseShift)
def _phase_shift(instruction) -> PhaseShift:
    return PhaseShift([instruction.target], instruction.angle)


class CPhaseShift(GateOperation):
    """Controlled phase shift gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * self._angle)]]
        )

    @property
    def targets(self) -> List[int]:
        return self._targets

    @property
    def control(self) -> int:
        return self._targets[0]


@from_braket_instruction.register(braket_instruction.CPhaseShift)
def _c_phase_shift(instruction) -> CPhaseShift:
    return CPhaseShift([instruction.control, instruction.target], instruction.angle)


class Identity(GateOperation):
    """Identity gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.eye(2)

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.I)
def _i(instruction) -> Identity:
    return Identity([instruction.target])


class RotX(GateOperation):
    """X-axis rotation gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_half_angle = np.cos(self._angle / 2)
        i_sin_half_angle = 1j * np.sin(self._angle / 2)
        return np.array([[cos_half_angle, -i_sin_half_angle], [-i_sin_half_angle, cos_half_angle]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Rx)
def _rot_x(instruction) -> RotX:
    return RotX([instruction.target], instruction.angle)


class RotY(GateOperation):
    """Y-axis rotation gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_half_angle = np.cos(self._angle / 2)
        sin_half_angle = np.sin(self._angle / 2)
        return np.array([[cos_half_angle, -sin_half_angle], [sin_half_angle, cos_half_angle]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Ry)
def _rot_y(instruction) -> RotY:
    return RotY([instruction.target], instruction.angle)


class RotZ(GateOperation):
    """Z-axis rotation gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        positive_phase = np.exp(1j * self._angle / 2)
        negative_phase = np.exp(-1j * self._angle / 2)
        return np.array([[negative_phase, 0], [0, positive_phase]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Rz)
def _rot_z(instruction) -> RotZ:
    return RotZ([instruction.target], instruction.angle)


class Swap(GateOperation):
    """Swap gate"""

    def __init__(self, targets):
        self._targets = targets

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Swap)
def _swap(instruction) -> Swap:
    return Swap(instruction.targets)


class XX(GateOperation):
    """Ising XX gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        positive_phase = np.exp(1j * self._angle)
        negative_phase = np.exp(-1j * self._angle)
        return np.array(
            [
                [1, 0, 0, -1j * positive_phase],
                [0, 1, -1j, 0],
                [0, -1j, 1, 0],
                [-1j * negative_phase, 0, 0, 1],
            ]
        ) / np.sqrt(2)

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.XX)
def _xx(instruction) -> XX:
    return XX(instruction.targets, instruction.angle)


class YY(GateOperation):
    """Ising YY gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        cos_angle = np.cos(self._angle)
        i_sin_angle = 1j * np.sin(self._angle)
        return np.array(
            [
                [cos_angle, 0, 0, i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [i_sin_angle, 0, 0, cos_angle],
            ]
        )

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.YY)
def _yy(instruction) -> YY:
    return YY(instruction.targets, instruction.angle)


class ZZ(GateOperation):
    """Ising ZZ gate"""

    def __init__(self, targets, angle):
        self._targets = targets
        self._angle = angle

    @property
    def matrix(self) -> np.ndarray:
        positive_phase = np.exp(1j * self._angle / 2)
        negative_phase = np.exp(-1j * self._angle / 2)
        return np.array(
            [
                [positive_phase, 0, 0, 0],
                [0, negative_phase, 0, 0],
                [0, 0, negative_phase, 0],
                [0, 0, 0, positive_phase],
            ]
        )

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.ZZ)
def _zz(instruction) -> ZZ:
    return ZZ(instruction.targets, instruction.angle)


class Unitary(GateOperation):
    """Unitary gate"""

    def __init__(self, targets, matrix):
        self._targets = targets
        self._matrix = matrix

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self._matrix)

    @property
    def targets(self) -> List[int]:
        return self._targets


@from_braket_instruction.register(braket_instruction.Unitary)
def _unitary(instruction) -> Unitary:
    def _from_ir_representation(matrix) -> np.ndarray:
        return np.array([[complex(element[0], element[1]) for element in row] for row in matrix])

    return Unitary(instruction.targets, _from_ir_representation(instruction.matrix))
