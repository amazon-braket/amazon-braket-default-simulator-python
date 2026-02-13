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

from braket.default_simulator.openqasm.simulation_path import FramedVariable, SimulationPath


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
        from unittest.mock import MagicMock

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
        from unittest.mock import MagicMock

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
