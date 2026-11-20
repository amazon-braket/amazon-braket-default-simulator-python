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

import pytest

from braket.default_simulator.density_matrix_simulator import DensityMatrixSimulator
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.state_vector_simulator import StateVectorSimulator
from braket.ir.openqasm import Program


@pytest.mark.parametrize("simulator_type", [StateVectorSimulator, DensityMatrixSimulator])
@pytest.mark.parametrize("control", ["break", "continue"])
@pytest.mark.parametrize("condition", ["b", "b == 1 && i == 1", "b == i"])
def test_static_for_control_restores_outer_variable(simulator_type, control, condition):
    source = f"""
    OPENQASM 3;
    qubit q;
    output bit b;
    output int i;
    i = 99;
    h q;
    b = measure q;
    for int i in [0:2] {{ if ({condition}) {{ {control}; }} }}
    """
    outputs = simulator_type().run_openqasm(Program(source=source), shots=128).outputs
    assert len(outputs) == 128
    assert all(row["i"] == 99 for row in outputs)

    context = simulator_type().create_program_context()
    context._shots = 128
    Interpreter(context).run(source)
    assert all(path.frame_number == 0 for path in context.active_paths)
    assert all(path.get_variable("i").value.value == 99 for path in context.active_paths)


@pytest.mark.parametrize("simulator_type", [StateVectorSimulator, DensityMatrixSimulator])
def test_nested_static_for_breaks_restore_both_scopes(simulator_type):
    source = """
    OPENQASM 3;
    qubit q;
    output bit b;
    output int i;
    output int j;
    i = 99;
    j = 77;
    h q;
    b = measure q;
    for int i in [0:2] {
        for int j in [0:2] {
            if (b == 1 && j == 1) { break; }
            if (b == 0 && j == 0) { continue; }
        }
        if (b == 1 && i == 1) { break; }
    }
    """
    outputs = simulator_type().run_openqasm(Program(source=source), shots=128).outputs
    assert len(outputs) == 128
    assert all(row["i"] == 99 and row["j"] == 77 for row in outputs)

    context = simulator_type().create_program_context()
    context._shots = 128
    Interpreter(context).run(source)
    assert all(path.frame_number == 0 for path in context.active_paths)
    assert all(path.get_variable("i").value.value == 99 for path in context.active_paths)
    assert all(path.get_variable("j").value.value == 77 for path in context.active_paths)


@pytest.mark.parametrize("simulator_type", [StateVectorSimulator, DensityMatrixSimulator])
@pytest.mark.parametrize("control", ["break", "continue"])
def test_function_while_condition_preserves_path_local_control(simulator_type, control):
    source = f"""
    OPENQASM 3;
    def below(int value) -> bool {{ return value < 3; }}
    qubit q;
    output bit b;
    output int count;
    output int iterations;
    count = 0;
    iterations = 0;
    h q;
    b = measure q;
    while (below(iterations)) {{
        iterations += 1;
        if (b) {{ {control}; }}
        count += 1;
    }}
    """
    outputs = simulator_type().run_openqasm(Program(source=source), shots=128).outputs
    assert len(outputs) == 128
    for row in outputs:
        assert row["count"] == (0 if row["b"] else 3)
        assert row["iterations"] == (1 if row["b"] and control == "break" else 3)


@pytest.mark.parametrize("simulator_type", [StateVectorSimulator, DensityMatrixSimulator])
def test_subroutine_return_after_static_for_break_restores_local(simulator_type):
    source = """
    OPENQASM 3;
    def choose(qubit p) -> int {
        bit b;
        int i = 99;
        b = measure p;
        for int i in [0:2] { if (b) { break; } }
        return i;
    }
    qubit q;
    h q;
    output int result;
    result = choose(q);
    """
    outputs = simulator_type().run_openqasm(Program(source=source), shots=128).outputs
    assert outputs == [{"result": 99}] * 128
