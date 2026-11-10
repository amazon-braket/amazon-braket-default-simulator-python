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

CHOOSE = """
def choose(qubit p) -> int {
    bit z;
    int r = 123;
    z = measure p;
    if (z == 0) { r = 7; } else { r = 9; }
    return r;
}
"""


@pytest.mark.parametrize("simulator_type", [StateVectorSimulator, DensityMatrixSimulator])
def test_initialized_local_return_after_measurement(simulator_type):
    source = "OPENQASM 3;" + CHOOSE + "qubit q; output int out; out = choose(q);"
    outputs = simulator_type().run_openqasm(Program(source=source), shots=16).outputs
    assert outputs == [{"out": 7}] * 16


@pytest.mark.parametrize("shots,batch_size", [(1, 1), (32, 1), (32, 16)])
@pytest.mark.parametrize(
    "declarations,statement,expected",
    [
        ("", "out = choose(q);", lambda m: 7 + 2 * m),
        ("", "int local = choose(q); out = local;", lambda m: 7 + 2 * m),
        (
            "def wrapper(qubit p) -> int { return choose(p) + 1; }",
            "out = wrapper(q);",
            lambda m: 8 + 2 * m,
        ),
        (
            "",
            "if (choose(q) == 7) { out = 11; } else { out = 13; }",
            lambda m: 11 + 2 * m,
        ),
        (
            "def read(qubit p) -> bit { bit b; b = measure p; return b; }",
            "out = int(read(q));",
            lambda m: m,
        ),
        (
            "def plus(int a) -> int { int r = a + 2; return r; }",
            "out = plus(choose(q));",
            lambda m: 9 + 2 * m,
        ),
        (
            "",
            "out = 0; for int i in [0:choose(q)] { out += 1; }",
            lambda m: 8 + 2 * m,
        ),
    ],
    ids=["assignment", "initializer", "nested", "condition", "bit", "argument", "range"],
)
def test_measurement_dependent_subroutine_returns(
    declarations, statement, expected, shots, batch_size
):
    source = f"""
    OPENQASM 3;
    {CHOOSE}
    {declarations}
    qubit q;
    U(pi/2, 0, pi) q;
    output int out;
    output bit m;
    output int r;
    r = 99;
    {statement}
    m = measure q;
    """
    result = StateVectorSimulator().run_openqasm(
        Program(source=source), shots=shots, batch_size=batch_size
    )
    assert len(result.outputs) == shots
    for output in result.outputs:
        assert output["out"] == expected(output["m"])
        assert output["r"] == 99


@pytest.mark.parametrize("shots", [1, 32])
def test_repeated_calls_and_operand_order(shots):
    source = """
    OPENQASM 3;
    def flip(qubit p) -> int {
        bit b;
        b = measure p;
        int r = int(b);
        U(pi, 0, pi) p;
        return r;
    }
    qubit q;
    U(pi/2, 0, pi) q;
    output bit initial;
    initial = measure q;
    output int out;
    out = flip(q) * 10 + flip(q);
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=shots).outputs
    assert len(outputs) == shots
    for output in outputs:
        assert output["out"] == 1 + 9 * output["initial"]


@pytest.mark.parametrize("lhs", ["", "a[0] + "])
def test_return_and_mutable_array_copy_back_have_same_path_value(lhs):
    source = (
        """
    OPENQASM 3;
    def write(mutable array[int[32], #dim=1] a, qubit p) -> int {
        bit b;
        b = measure p;
        if (b) { a[0] = 9; } else { a[0] = 7; }
        return a[0];
    }
    array[int[32], 1] a = {5};
    qubit q;
    U(pi/2, 0, pi) q;
    output int out;
    output int stored;
    out = """
        + lhs
        + """write(a, q);
    stored = a[0];
    """
    )
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=32).outputs
    assert len(outputs) == 32
    for output in outputs:
        assert output["out"] == output["stored"] + (5 if lhs else 0)
        assert output["stored"] in (7, 9)


def test_call_results_do_not_leak_into_registers_or_scopes():
    context = StateVectorSimulator().create_program_context()
    context._shots = 32
    interpreter = Interpreter(context)
    interpreter.run(
        "OPENQASM 3;"
        + CHOOSE
        + """
        qubit q;
        U(pi/2, 0, pi) q;
        output int out;
        choose(q);
        out = choose(q);
        """
    )
    assert interpreter._expression_temporaries == []
    assert not any(
        name.startswith("@call_result_")
        for scope in context.variable_table._scopes
        for name in scope
    )
    for path in context.active_paths:
        assert not any(name.startswith("@call_result_") for name in path.variables)
    assert not any(
        register.name.startswith("@call_result_")
        for register in context.circuit.classical_registers
    )
    assert list(context._output_variables) == ["out"]


@pytest.mark.parametrize(
    "gate", ["U(choose_angle(q[0]), 0, pi) q[1];", "ctrl @ gphase(choose_angle(q[0])) q[1];"]
)
def test_return_value_in_gate_angle(gate):
    # The controlled phase is observed by surrounding it with Hadamards.
    prepare = "U(pi/2, 0, pi) q[1];" if "gphase" in gate else ""
    source = (
        """
    OPENQASM 3;
    def choose_angle(qubit p) -> float {
        bit b;
        float a = 0;
        b = measure p;
        if (b) { a = pi; }
        return a;
    }
    qubit[2] q;
    U(pi/2, 0, pi) q[0];
    """
        + prepare
        + gate
        + prepare
        + """
    output bit[2] out;
    out = measure q;
    """
    )
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=32).outputs
    assert len(outputs) == 32
    assert all(output["out"][0] == output["out"][1] for output in outputs)


def test_call_in_while_condition_is_evaluated_once_per_iteration():
    source = """
    OPENQASM 3;
    def flip(qubit p) -> int {
        bit b;
        b = measure p;
        int r = int(b);
        U(pi, 0, pi) p;
        return r;
    }
    qubit q;
    U(pi/2, 0, pi) q;
    output bit initial;
    initial = measure q;
    output int count;
    count = 0;
    while (flip(q) == 0) { count += 1; }
    output bit final;
    final = measure q;
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=32).outputs
    assert len(outputs) == 32
    for output in outputs:
        assert output["count"] == 1 - output["initial"]
        assert output["final"] == 0


@pytest.mark.parametrize("body", ["count += 1;", "count += 1; continue;"])
def test_pure_call_in_while_condition_is_refreshed(body):
    context = Interpreter().run(
        """
        OPENQASM 3;
        def below(int a) -> bool { return a < 3; }
        int count = 0;
        while (below(count)) {
        """
        + body
        + "}"
    )
    assert context.get_value("count").value == 3


def test_pure_call_in_while_condition_keeps_iteration_limit(monkeypatch):
    monkeypatch.setattr("braket.default_simulator.openqasm.interpreter._MAX_LOOP_ITERATIONS", 3)
    with pytest.raises(RuntimeError, match="exceeded 3 iterations"):
        Interpreter().run("OPENQASM 3; def always() -> bool { return true; } while (always()) {}")


def test_call_in_reused_branch_body_runs_for_every_active_path():
    source = """
    OPENQASM 3;
    def read(qubit p) -> bit { bit b; b = measure p; return b; }
    qubit[2] q;
    U(pi/2, 0, pi) q[0];
    U(pi/2, 0, pi) q[1];
    output bit first;
    output bit second;
    output bit out;
    first = measure q[0];
    second = measure q[1];
    if (first == 0 || first == 1) { out = read(q[1]); }
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=128).outputs
    assert len(outputs) == 128
    assert all(output["out"] == output["second"] for output in outputs)


def test_while_call_condition_can_become_path_dependent_after_first_iteration():
    source = """
    OPENQASM 3;
    def positive(int value) -> bool { return value > 0; }
    qubit q;
    U(pi/2, 0, pi) q;
    output bit m;
    output int count;
    count = 0;
    int remaining = 2;
    while (positive(remaining)) {
        count += 1;
        if (remaining == 2) {
            m = measure q;
            remaining = int(m);
        } else {
            remaining = 0;
        }
    }
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=64).outputs
    assert len(outputs) == 64
    assert all(output["count"] == 1 + output["m"] for output in outputs)


def test_gate_arguments_preserve_values_before_later_call_side_effects():
    source = """
    OPENQASM 3;
    def write(mutable array[float[64], #dim=1] a) -> float {
        a[0] = pi;
        return 0;
    }
    array[float[64], 1] a = {0};
    qubit q;
    U(a[0], write(a), 0) q;
    output bit out;
    out = measure q;
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=32).outputs
    assert outputs == [{"out": 0}] * 32


@pytest.mark.parametrize(
    "statement,expected",
    [
        ("for int i in [a[0]:write(a)] { out += 1; }", 2),
        ("array[int[32], 2] v = {a[0], write(a)}; out = v[0] * 10 + v[1];", 1),
    ],
)
def test_ordered_value_operands_preserve_values_before_later_call(statement, expected):
    source = (
        """
    OPENQASM 3;
    def write(mutable array[int[32], #dim=1] a) -> int { a[0] = 3; return 1; }
    array[int[32], 1] a = {0};
    output int out;
    out = 0;
    """
        + statement
    )
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=16).outputs
    assert outputs == [{"out": expected}] * 16


def test_qubit_reference_index_is_evaluated_before_later_argument_call():
    source = """
    OPENQASM 3;
    def write(mutable array[int[32], #dim=1] a) -> int { a[0] = 1; return 0; }
    def use(qubit p, int ignored) { x p; }
    array[int[32], 1] a = {0};
    qubit[2] q;
    use(q[a[0]], write(a));
    output bit[2] out;
    out = measure q;
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=16).outputs
    assert outputs == [{"out": [1, 0]}] * 16


def test_gate_arguments_are_evaluated_before_qubit_indices():
    source = """
    OPENQASM 3;
    def write(mutable array[float[64], #dim=1] a) -> int { a[0] = pi; return 0; }
    array[float[64], 1] a = {0};
    qubit[1] q;
    U(a[0], 0, 0) q[write(a)];
    output bit out;
    out = measure q[0];
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=16).outputs
    assert outputs == [{"out": 0}] * 16


@pytest.mark.parametrize(
    "gate",
    ["pow(write(a)) @ U(a[0], 0, 0) q[0];", "pow(a[0]) @ x q[write(a) - 1];"],
)
def test_gate_modifiers_preserve_operand_order(gate):
    source = (
        """
    OPENQASM 3;
    def write(mutable array[float[64], #dim=1] a) -> int { a[0] = pi; return 1; }
    array[float[64], 1] a = {0};
    qubit[1] q;
    """
        + gate
        + """
    output bit out;
    out = measure q[0];
    """
    )
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=16).outputs
    assert outputs == [{"out": 0}] * 16


def test_gate_qubit_indices_are_evaluated_in_order():
    source = """
    OPENQASM 3;
    def write(mutable array[int[32], #dim=1] a) -> int { a[0] = 1; return 1; }
    array[int[32], 1] a = {0};
    qubit[2] q;
    x q[0];
    cnot q[a[0]], q[write(a)];
    output bit[2] out;
    out = measure q;
    """
    outputs = StateVectorSimulator().run_openqasm(Program(source=source), shots=16).outputs
    assert outputs == [{"out": [1, 1]}] * 16
