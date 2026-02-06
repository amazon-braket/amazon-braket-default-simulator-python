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

import braket.ir.jaqcd as instruction
import numpy as np
import pytest

from braket.default_simulator import gate_operations
from braket.default_simulator.gate_operations import Measure, Reset
from braket.default_simulator.operation_helpers import check_unitary, from_braket_instruction

testdata = [
    (instruction.I(target=4), (4,), gate_operations.Identity),
    (instruction.H(target=13), (13,), gate_operations.Hadamard),
    (instruction.X(target=11), (11,), gate_operations.PauliX),
    (instruction.Y(target=10), (10,), gate_operations.PauliY),
    (instruction.Z(target=9), (9,), gate_operations.PauliZ),
    (instruction.CNot(target=9, control=11), (11, 9), gate_operations.CX),
    (instruction.CV(target=9, control=11), (11, 9), gate_operations.CV),
    (instruction.CY(target=10, control=7), (7, 10), gate_operations.CY),
    (instruction.CZ(target=14, control=7), (7, 14), gate_operations.CZ),
    (instruction.ECR(targets=[4, 3]), (4, 3), gate_operations.ECR),
    (instruction.S(target=2), (2,), gate_operations.S),
    (instruction.Si(target=2), (2,), gate_operations.Si),
    (instruction.T(target=1), (1,), gate_operations.T),
    (instruction.Ti(target=1), (1,), gate_operations.Ti),
    (instruction.V(target=1), (1,), gate_operations.V),
    (instruction.Vi(target=1), (1,), gate_operations.Vi),
    (instruction.PhaseShift(target=2, angle=0.15), (2,), gate_operations.PhaseShift),
    (
        instruction.CPhaseShift(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift,
    ),
    (
        instruction.CPhaseShift00(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift00,
    ),
    (
        instruction.CPhaseShift01(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift01,
    ),
    (
        instruction.CPhaseShift10(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift10,
    ),
    (instruction.Rx(target=5, angle=0.14), (5,), gate_operations.RotX),
    (instruction.Ry(target=6, angle=0.16), (6,), gate_operations.RotY),
    (instruction.Rz(target=3, angle=0.17), (3,), gate_operations.RotZ),
    (instruction.Swap(targets=[4, 3]), (4, 3), gate_operations.Swap),
    (instruction.ISwap(targets=[4, 3]), (4, 3), gate_operations.ISwap),
    (instruction.PSwap(targets=[2, 1], angle=0.17), (2, 1), gate_operations.PSwap),
    (instruction.XY(targets=[2, 1], angle=0.17), (2, 1), gate_operations.XY),
    (instruction.XX(targets=[2, 1], angle=0.17), (2, 1), gate_operations.XX),
    (instruction.YY(targets=[2, 1], angle=0.17), (2, 1), gate_operations.YY),
    (instruction.ZZ(targets=[2, 1], angle=0.17), (2, 1), gate_operations.ZZ),
    (instruction.CCNot(target=9, controls=[13, 11]), (13, 11, 9), gate_operations.CCNot),
    (instruction.CSwap(targets=[9, 7], control=11), (11, 9, 7), gate_operations.CSwap),
    (
        instruction.Unitary(targets=[4], matrix=[[[0, 0], [0, 1]], [[1, 0], [0, 0]]]),
        (4,),
        gate_operations.Unitary,
    ),
]


@pytest.mark.parametrize("ir_instruction, targets, operation_type", testdata)
def test_gate_operation(ir_instruction, targets, operation_type):
    operation_instance = from_braket_instruction(ir_instruction)
    assert isinstance(operation_instance, operation_type)
    assert operation_instance.targets == targets
    check_unitary(operation_instance.matrix)


# ---------------------------------------------------------------------------
# Measure class tests
# ---------------------------------------------------------------------------


class TestMeasureBaseMatrix:
    """Cover all branches of Measure._base_matrix."""

    def test_identity_when_result_negative_one(self):
        m = Measure([0], result=-1)
        np.testing.assert_array_equal(m._base_matrix, np.eye(2))

    def test_project_to_zero(self):
        m = Measure([0], result=0)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_array_equal(m._base_matrix, expected)

    def test_project_to_one(self):
        m = Measure([0], result=1)
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        np.testing.assert_array_equal(m._base_matrix, expected)

    def test_invalid_result_returns_identity(self):
        m = Measure([0], result=99)
        np.testing.assert_array_equal(m._base_matrix, np.eye(2))


class TestMeasureApply:
    """Cover Measure.apply() projection and normalization."""

    def test_apply_no_op_when_unset(self):
        m = Measure([0], result=-1)
        state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        result = m.apply(state)
        np.testing.assert_array_almost_equal(result, state)

    def test_apply_project_to_zero(self):
        # |+⟩ = (|0⟩ + |1⟩)/√2  →  project to |0⟩  →  |0⟩
        m = Measure([0], result=0)
        state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        result = m.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([1, 0], dtype=complex))

    def test_apply_project_to_one(self):
        # |+⟩ = (|0⟩ + |1⟩)/√2  →  project to |1⟩  →  |1⟩
        m = Measure([0], result=1)
        state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        result = m.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([0, 1], dtype=complex))

    def test_apply_two_qubit_state(self):
        # Bell state (|00⟩ + |11⟩)/√2, measure qubit 0 → 0 → collapses to |00⟩
        m = Measure([0], result=0)
        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        result = m.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([1, 0, 0, 0], dtype=complex))

    def test_apply_zero_norm_state(self):
        # Edge case: state already zero in the projected subspace
        m = Measure([0], result=0)
        state = np.array([0, 1], dtype=complex)  # |1⟩
        result = m.apply(state)
        # All zeros, norm=0, should return zeros without dividing by zero
        np.testing.assert_array_almost_equal(result, np.array([0, 0], dtype=complex))

    def test_apply_multi_target_passthrough(self):
        # Measure with 2 targets — the single-qubit branch is skipped, state returned as-is
        m = Measure([0, 1], result=0)
        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        result = m.apply(state)
        # No projection applied for multi-target, just returns the copy
        np.testing.assert_array_almost_equal(result, state)


# ---------------------------------------------------------------------------
# Reset class tests
# ---------------------------------------------------------------------------


class TestResetApply:
    """Cover Reset._base_matrix and Reset.apply()."""

    def test_base_matrix_is_identity(self):
        r = Reset([0])
        np.testing.assert_array_equal(r._base_matrix, np.eye(2))

    def test_reset_qubit_in_one_state(self):
        # |1⟩ → reset → |0⟩
        r = Reset([0])
        state = np.array([0, 1], dtype=complex)
        result = r.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([1, 0], dtype=complex))

    def test_reset_qubit_already_zero(self):
        # |0⟩ → reset → |0⟩ (no change)
        r = Reset([0])
        state = np.array([1, 0], dtype=complex)
        result = r.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([1, 0], dtype=complex))

    def test_reset_superposition(self):
        # (|0⟩ + |1⟩)/√2 → reset → |0⟩ (all amplitude transferred to |0⟩)
        r = Reset([0])
        state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        result = r.apply(state)
        assert abs(result[0]) > 0.99
        assert abs(result[1]) < 1e-10

    def test_reset_second_qubit_in_two_qubit_system(self):
        # |01⟩ → reset qubit 1 → |00⟩
        r = Reset([1])
        state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        result = r.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([1, 0, 0, 0], dtype=complex))

    def test_reset_multi_target_passthrough(self):
        # Reset with 2 targets — the single-qubit branch is skipped
        r = Reset([0, 1])
        state = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        result = r.apply(state)
        # No reset applied for multi-target, state returned as-is
        np.testing.assert_array_almost_equal(result, state)

    def test_reset_zero_norm_state(self):
        # Edge case: all-zero state — norm is 0, should not divide by zero
        r = Reset([0])
        state = np.array([0, 0], dtype=complex)
        result = r.apply(state)
        np.testing.assert_array_almost_equal(result, np.array([0, 0], dtype=complex))
