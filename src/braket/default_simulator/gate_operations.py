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

from collections.abc import Sequence

import numpy as np

import braket.ir.jaqcd as braket_instruction
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.operation_helpers import (
    _from_braket_instruction,
    check_matrix_dimensions,
    check_unitary,
    ir_matrix_to_ndarray,
)


class Identity(GateOperation):
    """Identity gate"""

    _matrix = np.eye(2, dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return Identity._matrix

    @property
    def gate_type(self) -> str:
        return "identity"


@_from_braket_instruction.register(braket_instruction.I)
def _i(instruction) -> Identity:
    return Identity([instruction.target])


class Hadamard(GateOperation):
    """Hadamard gate"""

    _matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return Hadamard._matrix

    @property
    def gate_type(self) -> str:
        return "hadamard"


@_from_braket_instruction.register(braket_instruction.H)
def _hadamard(instruction) -> Hadamard:
    return Hadamard([instruction.target])


class PauliX(GateOperation):
    """Pauli-X gate"""

    _matrix = np.array([[0, 1], [1, 0]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return PauliX._matrix

    @property
    def gate_type(self) -> str:
        return "pauli_x"


@_from_braket_instruction.register(braket_instruction.X)
def _pauli_x(instruction) -> PauliX:
    return PauliX([instruction.target])


class PauliY(GateOperation):
    """Pauli-Y gate"""

    _matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return PauliY._matrix

    @property
    def gate_type(self) -> str:
        return "pauli_y"


@_from_braket_instruction.register(braket_instruction.Y)
def _pauli_y(instruction) -> PauliY:
    return PauliY([instruction.target])


class PauliZ(GateOperation):
    """Pauli-Z gate"""

    _matrix = np.diag([1.0, -1.0]).astype(complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return PauliZ._matrix

    @property
    def gate_type(self) -> str:
        return "pauli_z"


@_from_braket_instruction.register(braket_instruction.Z)
def _pauli_z(instruction) -> PauliZ:
    return PauliZ([instruction.target])


class CV(GateOperation):
    """Controlled-Sqrt(NOT) gate"""

    _matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5 + 0.5j, 0.5 - 0.5j],
            [0, 0, 0.5 - 0.5j, 0.5 + 0.5j],
        ],
        dtype=complex,
    )

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return CV._matrix

    @property
    def gate_type(self) -> str:
        return "cv"


@_from_braket_instruction.register(braket_instruction.CV)
def _cv(instruction) -> CV:
    return CV([instruction.control, instruction.target])


class CX(GateOperation):
    """Controlled Pauli-X gate"""

    _matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return CX._matrix

    @property
    def gate_type(self) -> str:
        return "cx"


@_from_braket_instruction.register(braket_instruction.CNot)
def _cx(instruction) -> CX:
    return CX([instruction.control, instruction.target])


class CY(GateOperation):
    """Controlled Pauli-Y gate"""

    _matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self):
        return CY._matrix

    @property
    def gate_type(self) -> str:
        return "cy"


@_from_braket_instruction.register(braket_instruction.CY)
def _cy(instruction) -> CY:
    return CY([instruction.control, instruction.target])


class CZ(GateOperation):
    """Controlled Pauli-Z gate"""

    _matrix = np.diag([1, 1, 1, -1]).astype(complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return CZ._matrix

    @property
    def gate_type(self) -> str:
        return "cz"


@_from_braket_instruction.register(braket_instruction.CZ)
def _cz(instruction) -> CZ:
    return CZ([instruction.control, instruction.target])


class ECR(GateOperation):
    """ECR gate"""

    _matrix = (
        1
        / np.sqrt(2)
        * np.array(
            [[0, 0, 1, 1.0j], [0, 0, 1.0j, 1], [1, -1.0j, 0, 0], [-1.0j, 1, 0, 0]],
            dtype=complex,
        )
    )

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return ECR._matrix

    @property
    def gate_type(self) -> str:
        return "ecr"


@_from_braket_instruction.register(braket_instruction.ECR)
def _ecr(instruction) -> ECR:
    return ECR(instruction.targets)


class S(GateOperation):
    """S gate"""

    _matrix = np.array([[1, 0], [0, 1j]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return S._matrix

    @property
    def gate_type(self) -> str:
        return "s"


@_from_braket_instruction.register(braket_instruction.S)
def _s(instruction) -> S:
    return S([instruction.target])


class Si(GateOperation):
    r"""The adjoint :math:`S^{\dagger}` of the S gate"""

    _matrix = np.array([[1, 0], [0, -1j]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return Si._matrix

    @property
    def gate_type(self) -> str:
        return "si"


@_from_braket_instruction.register(braket_instruction.Si)
def _si(instruction) -> Si:
    return Si([instruction.target])


class T(GateOperation):
    """T gate"""

    _matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return T._matrix

    @property
    def gate_type(self) -> str:
        return "t"


@_from_braket_instruction.register(braket_instruction.T)
def _t(instruction) -> T:
    return T([instruction.target])


class Ti(GateOperation):
    r"""The adjoint :math:`T^{\dagger}` of the T gate"""

    _matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return Ti._matrix

    @property
    def gate_type(self) -> str:
        return "ti"


@_from_braket_instruction.register(braket_instruction.Ti)
def _ti(instruction) -> Ti:
    return Ti([instruction.target])


class V(GateOperation):
    """Square root of the X (not) gate"""

    _matrix = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return V._matrix

    @property
    def gate_type(self) -> str:
        return "v"


@_from_braket_instruction.register(braket_instruction.V)
def _v(instruction) -> V:
    return V([instruction.target])


class Vi(GateOperation):
    r"""The adjoint :math:`V^{\dagger}` of the square root of the X (not) gate"""

    _matrix = np.array(([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]), dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return Vi._matrix

    @property
    def gate_type(self) -> str:
        return "vi"


@_from_braket_instruction.register(braket_instruction.Vi)
def _vi(instruction) -> Vi:
    return Vi([instruction.target])


class PhaseShift(GateOperation):
    """Phase shift gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        return np.array([[1, 0], [0, np.exp(1j * angle_float)]], dtype=complex)

    @property
    def gate_type(self) -> str:
        return "phaseshift"


@_from_braket_instruction.register(braket_instruction.PhaseShift)
def _phase_shift(instruction) -> PhaseShift:
    return PhaseShift([instruction.target], instruction.angle)


class CPhaseShift(GateOperation):
    """Controlled phase shift gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        return np.diag([1.0, 1.0, 1.0, np.exp(1j * angle_float)]).astype(complex)

    @property
    def gate_type(self) -> str:
        return "cphaseshift"


@_from_braket_instruction.register(braket_instruction.CPhaseShift)
def _c_phase_shift(instruction) -> CPhaseShift:
    return CPhaseShift([instruction.control, instruction.target], instruction.angle)


class CPhaseShift00(GateOperation):
    r"""Controlled phase shift gate phasing the :math:`\ket{00}` state"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        return np.diag([np.exp(1j * angle_float), 1.0, 1.0, 1.0]).astype(complex)

    @property
    def gate_type(self) -> str:
        return "cphaseshift00"


@_from_braket_instruction.register(braket_instruction.CPhaseShift00)
def _c_phase_shift_00(instruction) -> CPhaseShift00:
    return CPhaseShift00([instruction.control, instruction.target], instruction.angle)


class CPhaseShift01(GateOperation):
    r"""Controlled phase shift gate phasing the :math:`\ket{01}` state"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        return np.diag([1.0, np.exp(1j * angle_float), 1.0, 1.0]).astype(complex)

    @property
    def gate_type(self) -> str:
        return "cphaseshift01"


@_from_braket_instruction.register(braket_instruction.CPhaseShift01)
def _c_phase_shift_01(instruction) -> CPhaseShift01:
    return CPhaseShift01([instruction.control, instruction.target], instruction.angle)


class CPhaseShift10(GateOperation):
    r"""Controlled phase shift gate phasing the :math:`\ket{10}` state"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        return np.diag([1.0, 1.0, np.exp(1j * angle_float), 1.0]).astype(complex)

    @property
    def gate_type(self) -> str:
        return "cphaseshift10"


@_from_braket_instruction.register(braket_instruction.CPhaseShift10)
def _c_phase_shift_10(instruction) -> CPhaseShift10:
    return CPhaseShift10([instruction.control, instruction.target], instruction.angle)


class RotX(GateOperation):
    """X-axis rotation gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        cos_half_angle = np.cos(angle_float / 2).astype(complex)
        i_sin_half_angle = 1j * np.sin(angle_float / 2).astype(complex)
        return np.array(
            [[cos_half_angle, -i_sin_half_angle], [-i_sin_half_angle, cos_half_angle]],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "rx"


@_from_braket_instruction.register(braket_instruction.Rx)
def _rot_x(instruction) -> RotX:
    return RotX([instruction.target], instruction.angle)


class RotY(GateOperation):
    """Y-axis rotation gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        cos_half_angle = np.cos(angle_float / 2).astype(complex)
        sin_half_angle = np.sin(angle_float / 2).astype(complex)
        return np.array(
            [[cos_half_angle, -sin_half_angle], [sin_half_angle, cos_half_angle]], dtype=complex
        )

    @property
    def gate_type(self) -> str:
        return "ry"


@_from_braket_instruction.register(braket_instruction.Ry)
def _rot_y(instruction) -> RotY:
    return RotY([instruction.target], instruction.angle)


class RotZ(GateOperation):
    """Z-axis rotation gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        return np.array(
            [[np.exp(-1j * angle_float / 2), 0], [0, np.exp(1j * angle_float / 2)]], dtype=complex
        )

    @property
    def gate_type(self) -> str:
        return "rz"


@_from_braket_instruction.register(braket_instruction.Rz)
def _rot_z(instruction) -> RotZ:
    return RotZ([instruction.target], instruction.angle)


class Swap(GateOperation):
    """Swap gate"""

    _matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return Swap._matrix

    @property
    def gate_type(self) -> str:
        return "swap"


@_from_braket_instruction.register(braket_instruction.Swap)
def _swap(instruction) -> Swap:
    return Swap(instruction.targets)


class ISwap(GateOperation):
    """iSwap gate"""

    _matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0j, 0.0],
            [0.0, 1.0j, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return ISwap._matrix

    @property
    def gate_type(self) -> str:
        return "iswap"


@_from_braket_instruction.register(braket_instruction.ISwap)
def _iswap(instruction) -> ISwap:
    return ISwap(instruction.targets)


class PSwap(GateOperation):
    """Parametrized Swap gate"""

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        exp_angle = np.exp(1j * angle_float)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, exp_angle, 0.0],
                [0.0, exp_angle, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "pswap"


@_from_braket_instruction.register(braket_instruction.PSwap)
def _pswap(instruction) -> PSwap:
    return PSwap(instruction.targets, instruction.angle)


class XY(GateOperation):
    """XY gate

    Reference: https://arxiv.org/abs/1912.04424v1
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        cos = np.cos(angle_float / 2).astype(complex)
        sin = np.sin(angle_float / 2).astype(complex)
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
    def gate_type(self) -> str:
        return "xy"


@_from_braket_instruction.register(braket_instruction.XY)
def _xy(instruction) -> XY:
    return XY(instruction.targets, instruction.angle)


class XX(GateOperation):
    """Ising XX gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        cos_angle = np.cos(angle_float / 2).astype(complex)
        i_sin_angle = 1j * np.sin(angle_float / 2).astype(complex)
        return np.array(
            [
                [cos_angle, 0, 0, -i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [-i_sin_angle, 0, 0, cos_angle],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "xx"


@_from_braket_instruction.register(braket_instruction.XX)
def _xx(instruction) -> XX:
    return XX(instruction.targets, instruction.angle)


class YY(GateOperation):
    """Ising YY gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        cos_angle = np.cos(angle_float / 2).astype(complex)
        i_sin_angle = 1j * np.sin(angle_float / 2).astype(complex)
        return np.array(
            [
                [cos_angle, 0, 0, i_sin_angle],
                [0, cos_angle, -i_sin_angle, 0],
                [0, -i_sin_angle, cos_angle, 0],
                [i_sin_angle, 0, 0, cos_angle],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "yy"


@_from_braket_instruction.register(braket_instruction.YY)
def _yy(instruction) -> YY:
    return YY(instruction.targets, instruction.angle)


class ZZ(GateOperation):
    """Ising ZZ gate

    Reference: https://arxiv.org/abs/1707.06356
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        positive_phase = np.exp(1j * angle_float / 2)
        negative_phase = np.exp(-1j * angle_float / 2)
        return np.array(
            [
                [negative_phase, 0, 0, 0],
                [0, positive_phase, 0, 0],
                [0, 0, positive_phase, 0],
                [0, 0, 0, negative_phase],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "zz"


@_from_braket_instruction.register(braket_instruction.ZZ)
def _zz(instruction) -> ZZ:
    return ZZ(instruction.targets, instruction.angle)


class CCNot(GateOperation):
    """Controlled CNot or Toffoli gate"""

    _matrix = np.array(
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

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return CCNot._matrix

    @property
    def gate_type(self) -> str:
        return "ccnot"


@_from_braket_instruction.register(braket_instruction.CCNot)
def _ccnot(instruction) -> CCNot:
    return CCNot([*instruction.controls, instruction.target])


class CSwap(GateOperation):
    """Controlled Swap gate"""

    _matrix = np.array(
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

    def __init__(self, targets, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )

    @property
    def _base_matrix(self) -> np.ndarray:
        return CSwap._matrix

    @property
    def gate_type(self) -> str:
        return "cswap"


@_from_braket_instruction.register(braket_instruction.CSwap)
def _cswap(instruction) -> CSwap:
    return CSwap([instruction.control, *instruction.targets])


class PRx(GateOperation):
    """
    PhaseRx gate.
    """

    def __init__(self, targets, angle_1, angle_2, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle_1 = angle_1
        self._angle_2 = angle_2

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_1_float = float(self._angle_1)
        angle_2_float = float(self._angle_2)
        cos = np.cos(angle_1_float / 2)
        isin = -1j * np.sin(angle_1_float / 2)
        exp_neg_angle_2 = np.exp(-1j * angle_2_float)
        exp_pos_angle_2 = np.exp(1j * angle_2_float)
        return np.array(
            [
                [
                    cos,
                    exp_neg_angle_2 * isin,
                ],
                [
                    exp_pos_angle_2 * isin,
                    cos,
                ],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "prx"


class GPi(GateOperation):
    """
    IonQ GPi gate.

    Reference: https://ionq.com/docs/getting-started-with-native-gates#single-qubit-gates
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        exp_neg_angle = np.exp(-1j * angle_float)
        exp_pos_angle = np.exp(1j * angle_float)
        return np.array(
            [
                [0, exp_neg_angle],
                [exp_pos_angle, 0],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "gpi"


class GPi2(GateOperation):
    """
    IonQ GPi2 gate.

    Reference: https://ionq.com/docs/getting-started-with-native-gates#single-qubit-gates
    """

    def __init__(self, targets, angle, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle = angle

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_float = float(self._angle)
        exp_neg_angle = np.exp(-1j * angle_float)
        exp_pos_angle = np.exp(1j * angle_float)
        return np.array(
            [
                [1, -1j * exp_neg_angle],
                [-1j * exp_pos_angle, 1],
            ],
            dtype=complex,
        ) / np.sqrt(2)

    @property
    def gate_type(self) -> str:
        return "gpi2"


class MS(GateOperation):
    """
    IonQ MS gate.

    Reference: https://ionq.com/docs/getting-started-with-native-gates#entangling-gates
    """

    def __init__(self, targets, angle_1, angle_2, angle_3=np.pi / 2, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._angle_1 = angle_1
        self._angle_2 = angle_2
        self._angle_3 = angle_3

    @property
    def _base_matrix(self) -> np.ndarray:
        angle_1_float = float(self._angle_1)
        angle_2_float = float(self._angle_2)
        angle_3_float = float(self._angle_3)
        cos = np.cos(angle_3_float / 2)
        isin = -1j * np.sin(angle_3_float / 2)
        exp_neg_sum = np.exp(-1j * (angle_1_float + angle_2_float))
        exp_neg_diff = np.exp(-1j * (angle_1_float - angle_2_float))
        exp_pos_diff = np.exp(1j * (angle_1_float - angle_2_float))
        exp_pos_sum = np.exp(1j * (angle_1_float + angle_2_float))
        return np.array(
            [
                [
                    cos,
                    0,
                    0,
                    exp_neg_sum * isin,
                ],
                [
                    0,
                    cos,
                    exp_neg_diff * isin,
                    0,
                ],
                [
                    0,
                    exp_pos_diff * isin,
                    cos,
                    0,
                ],
                [
                    exp_pos_sum * isin,
                    0,
                    0,
                    cos,
                ],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "ms"


class Unitary(GateOperation):
    """Arbitrary unitary gate"""

    def __init__(self, targets, matrix, ctrl_modifiers=(), power=1):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        clone = np.array(matrix, dtype=complex)
        check_matrix_dimensions(clone, self._targets)
        check_unitary(clone)
        self._matrix = clone

    @property
    def _base_matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def gate_type(self) -> str:
        return "unitary"


@_from_braket_instruction.register(braket_instruction.Unitary)
def _unitary(instruction) -> Unitary:
    return Unitary(instruction.targets, ir_matrix_to_ndarray(instruction.matrix))


"""
OpenQASM gate operations
"""


class U(GateOperation):
    """
    Parameterized primitive gate for OpenQASM simulator
    """

    def __init__(
        self,
        targets: Sequence[int],
        theta: float,
        phi: float,
        lambda_: float,
        ctrl_modifiers: Sequence[int] = (),
        power: float = 1,
    ):
        super().__init__(
            targets=targets,
            ctrl_modifiers=ctrl_modifiers,
            power=power,
        )
        self._theta = theta
        self._phi = phi
        self._lambda = lambda_

    @property
    def _base_matrix(self) -> np.ndarray:
        """
        Generate parameterized Unitary matrix.
        https://openqasm.com/language/gates.html#built-in-gates

        Returns:
            np.ndarray: U Matrix
        """
        theta_float = float(self._theta)
        phi_float = float(self._phi)
        lambda_float = float(self._lambda)
        cos = np.cos(theta_float / 2).astype(complex)
        sin = np.sin(theta_float / 2).astype(complex)
        exp_lambda = np.exp(1j * lambda_float)
        exp_phi = np.exp(1j * phi_float)
        exp_phi_lambda = np.exp(1j * (phi_float + lambda_float))
        return np.array(
            [
                [cos, -exp_lambda * sin],
                [exp_phi * sin, exp_phi_lambda * cos],
            ],
            dtype=complex,
        )

    @property
    def gate_type(self) -> str:
        return "u"


class GPhase(GateOperation):
    """
    Global phase operation for OpenQASM simulator
    """

    def __init__(self, targets: Sequence[int], angle: float):
        super().__init__(
            targets=targets,
        )
        angle_float = float(angle)
        self._exp = np.exp(angle_float * 1j) * np.eye(2 ** len(self._targets), dtype=complex)

    @property
    def _base_matrix(self) -> np.ndarray:
        return self._exp

    @property
    def gate_type(self) -> str:
        return "gphase"


BRAKET_GATES = {
    "i": Identity,
    "h": Hadamard,
    "x": PauliX,
    "y": PauliY,
    "z": PauliZ,
    "cv": CV,
    "cnot": CX,
    "cy": CY,
    "cz": CZ,
    "ecr": ECR,
    "s": S,
    "si": Si,
    "t": T,
    "ti": Ti,
    "v": V,
    "vi": Vi,
    "phaseshift": PhaseShift,
    "cphaseshift": CPhaseShift,
    "cphaseshift00": CPhaseShift00,
    "cphaseshift01": CPhaseShift01,
    "cphaseshift10": CPhaseShift10,
    "rx": RotX,
    "ry": RotY,
    "rz": RotZ,
    "swap": Swap,
    "iswap": ISwap,
    "pswap": PSwap,
    "xy": XY,
    "xx": XX,
    "yy": YY,
    "zz": ZZ,
    "ccnot": CCNot,
    "cswap": CSwap,
    "prx": PRx,
    "gpi": GPi,
    "gpi2": GPi2,
    "ms": MS,
    "unitary": Unitary,
    "U": U,
    "gphase": GPhase,
}
