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

from unittest.mock import MagicMock

import numpy as np

from braket.default_simulator.openqasm.simulation_path import (
    FramedVariable,
    SimulationPath,
    SubEnsemble,
)


class TestFramedVariable:
    def test_init_and_properties(self):
        var = FramedVariable(name="x", var_type=int, value=42, is_const=False, frame_number=1)
        assert var.name == "x"
        assert var.var_type is int
        assert var.value == 42
        assert var.is_const is False
        assert var.frame_number == 1

    def test_const_variable(self):
        var = FramedVariable(name="PI", var_type=float, value=3.14, is_const=True, frame_number=0)
        assert var.is_const is True

    def test_value_setter(self):
        var = FramedVariable(name="x", var_type=int, value=0, is_const=False, frame_number=0)
        var.value = 99
        assert var.value == 99


class TestSimulationPath:
    def test_default_init(self):
        path = SimulationPath()
        assert path.instructions == []
        assert path.shots == 0
        assert path.variables == {}
        assert path.measurements == {}
        assert path.frame_number == 0

    def test_init_with_values(self):
        var = FramedVariable("x", int, 5, False, 0)
        path = SimulationPath(
            instructions=[],
            shots=100,
            variables={"x": var},
            measurements={0: [1]},
            frame_number=2,
        )
        assert path.shots == 100
        assert path.variables["x"].value == 5
        assert path.measurements == {0: [1]}
        assert path.frame_number == 2

    def test_shots_setter(self):
        path = SimulationPath(shots=100)
        path.shots = 50
        assert path.shots == 50

    def test_frame_number_setter(self):
        path = SimulationPath(frame_number=0)
        path.frame_number = 3
        assert path.frame_number == 3

    def test_add_instruction(self):
        """Test that instructions are appended correctly."""
        path = SimulationPath()
        mock_op = MagicMock()
        path.add_instruction(mock_op)
        assert len(path.instructions) == 1
        assert path.instructions[0] is mock_op

    def test_set_and_get_variable(self):
        path = SimulationPath()
        var = FramedVariable("y", float, 3.14, False, 0)
        path.set_variable("y", var)
        retrieved = path.get_variable("y")
        assert retrieved is var

    def test_get_variable_missing(self):
        path = SimulationPath()
        assert path.get_variable("nonexistent") is None

    def test_record_measurement(self):
        path = SimulationPath()
        path.record_measurement(0, 1)
        path.record_measurement(0, 0)
        path.record_measurement(1, 1)
        assert path.measurements == {0: [1, 0], 1: [1]}

    def test_branch_creates_independent_copy(self):
        """Validates: Requirements 7.1 - deep-copy variable state on branch."""
        var = FramedVariable("x", int, 10, False, 0)
        parent = SimulationPath(
            instructions=[],
            shots=100,
            variables={"x": var},
            measurements={0: [1]},
            frame_number=1,
        )
        child = parent.branch()

        # Child has same values
        assert child.shots == 100
        assert child.variables["x"].value == 10
        assert child.measurements == {0: [1]}
        assert child.frame_number == 1

        # Modifying child does not affect parent
        child.shots = 50
        child.variables["x"].value = 99
        child.record_measurement(0, 0)

        assert parent.shots == 100
        assert parent.variables["x"].value == 10
        assert parent.measurements == {0: [1]}

    def test_branch_instructions_independent(self):
        """Instructions list is independent after branching."""
        parent = SimulationPath(instructions=[MagicMock()])
        child = parent.branch()

        child.add_instruction(MagicMock())
        assert len(parent.instructions) == 1
        assert len(child.instructions) == 2

    def test_enter_frame(self):
        path = SimulationPath(frame_number=0)
        prev = path.enter_frame()
        assert prev == 0
        assert path.frame_number == 1

    def test_exit_frame_removes_newer_variables(self):
        """Validates: Requirements 7.3 - scope restoration."""
        path = SimulationPath(frame_number=0)
        path.set_variable("outer", FramedVariable("outer", int, 1, False, 0))

        prev = path.enter_frame()
        path.set_variable("inner", FramedVariable("inner", int, 2, False, 1))
        assert "inner" in path.variables

        path.exit_frame(prev)
        assert "outer" in path.variables
        assert "inner" not in path.variables
        assert path.frame_number == 0

    def test_nested_frames(self):
        """Test multiple nested scope frames."""
        path = SimulationPath(frame_number=0)
        path.set_variable("a", FramedVariable("a", int, 1, False, 0))

        frame0 = path.enter_frame()  # frame 1
        path.set_variable("b", FramedVariable("b", int, 2, False, 1))

        frame1 = path.enter_frame()  # frame 2
        path.set_variable("c", FramedVariable("c", int, 3, False, 2))

        assert set(path.variables.keys()) == {"a", "b", "c"}

        path.exit_frame(frame1)
        assert set(path.variables.keys()) == {"a", "b"}

        path.exit_frame(frame0)
        assert set(path.variables.keys()) == {"a"}


class TestSubEnsemble:
    @staticmethod
    def _single_qubit_dm(p0: float) -> np.ndarray:
        """Diagonal single-qubit density matrix with population p0 in |0>."""
        return np.array([[p0, 0.0], [0.0, 1.0 - p0]], dtype=complex)

    def test_default_init(self):
        sub = SubEnsemble()
        assert sub.density_matrix is None
        assert sub.variables == {}
        assert sub.measurements == {}
        assert sub.frame_number == 0

    def test_is_simulation_path(self):
        """SubEnsemble retains the inherited SimulationPath interface."""
        sub = SubEnsemble(density_matrix=self._single_qubit_dm(0.5))
        assert isinstance(sub, SimulationPath)

    def test_init_with_classical_state(self):
        var = FramedVariable("x", int, 5, False, 0)
        sub = SubEnsemble(
            density_matrix=self._single_qubit_dm(0.25),
            variables={"x": var},
            measurements={0: [1]},
            frame_number=2,
        )
        assert sub.variables["x"].value == 5
        assert sub.measurements == {0: [1]}
        assert sub.frame_number == 2

    def test_trace_reflects_matrix(self):
        """Validates: Requirements 1.5 - trace equals the matrix trace."""
        sub = SubEnsemble(density_matrix=self._single_qubit_dm(0.5))
        assert np.isclose(sub.trace, 1.0)

    def test_trace_of_unnormalized_matrix(self):
        """Validates: Requirements 1.5 - trace of an unnormalized branch."""
        # An unnormalized branch with trace 0.3 (joint probability of its tag).
        sub = SubEnsemble(density_matrix=np.array([[0.3, 0.0], [0.0, 0.0]], dtype=complex))
        assert np.isclose(sub.trace, 0.3)

    def test_trace_is_real_float(self):
        """Trace is returned as a real Python float even for complex matrices."""
        rho = np.array([[0.5, 0.2j], [-0.2j, 0.5]], dtype=complex)
        sub = SubEnsemble(density_matrix=rho)
        assert isinstance(sub.trace, float)
        assert np.isclose(sub.trace, 1.0)

    def test_branch_yields_independent_matrix(self):
        """Validates: Requirements 1.5 - branch copies the density matrix."""
        rho = self._single_qubit_dm(0.5)
        parent = SubEnsemble(density_matrix=rho)
        child = parent.branch()

        assert isinstance(child, SubEnsemble)
        assert np.allclose(child.density_matrix, parent.density_matrix)

        # Mutating the child matrix must not affect the parent.
        child.density_matrix[0, 0] = 0.9
        assert parent.density_matrix[0, 0] == 0.5
        # The original source array is also untouched (branch copies).
        assert rho[0, 0] == 0.5

    def test_branch_yields_independent_classical_state(self):
        """Validates: Requirements 9.6, 9.7 - branch deep-copies classical state."""
        var = FramedVariable("x", int, 10, False, 0)
        parent = SubEnsemble(
            density_matrix=self._single_qubit_dm(0.5),
            variables={"x": var},
            measurements={0: [1]},
            frame_number=1,
        )
        child = parent.branch()

        assert child.variables["x"].value == 10
        assert child.measurements == {0: [1]}
        assert child.frame_number == 1

        # Modifying child classical state does not affect parent.
        child.variables["x"].value = 99
        child.record_measurement(0, 0)
        child.set_variable("y", FramedVariable("y", int, 7, False, 1))

        assert parent.variables["x"].value == 10
        assert parent.measurements == {0: [1]}
        assert "y" not in parent.variables

    def test_branch_preserves_trace(self):
        """A branched copy has the same trace as its parent."""
        sub = SubEnsemble(density_matrix=np.array([[0.4, 0.0], [0.0, 0.0]], dtype=complex))
        child = sub.branch()
        assert np.isclose(child.trace, sub.trace)

    def test_inherited_frame_scoping(self):
        """Validates: Requirements 9.7 - inherited frame scoping still works."""
        sub = SubEnsemble(density_matrix=self._single_qubit_dm(0.5))
        sub.set_variable("outer", FramedVariable("outer", int, 1, False, 0))

        prev = sub.enter_frame()
        assert sub.frame_number == 1
        sub.set_variable("inner", FramedVariable("inner", int, 2, False, 1))
        assert "inner" in sub.variables

        sub.exit_frame(prev)
        assert "outer" in sub.variables
        assert "inner" not in sub.variables
        assert sub.frame_number == 0

    def test_inherited_record_measurement(self):
        """Validates: Requirements 9.6 - inherited measurement recording works."""
        sub = SubEnsemble(density_matrix=self._single_qubit_dm(0.5))
        sub.record_measurement(0, 1)
        sub.record_measurement(0, 0)
        sub.record_measurement(2, 1)
        assert sub.measurements == {0: [1, 0], 2: [1]}
