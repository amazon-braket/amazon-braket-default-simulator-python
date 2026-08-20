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

from collections import Counter

import numpy as np
import pytest

from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.output_helpers import (
    compute_outputs_branched,
    compute_outputs_single_path,
    convert_output_value,
)
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    AngleType,
    BitType,
    BoolType,
    FloatType,
    IntegerLiteral,
    IntType,
    UintType,
)
from braket.default_simulator.state_vector_simulator import StateVectorSimulator
from braket.ir.openqasm import Program as OpenQASMProgram


def _shots(context, shots):
    context._shots = shots
    return context


def run_program(qasm, shots=0):
    """Interpret a program with a simulator-backed context."""
    context = _shots(StateVectorSimulator().create_program_context(), shots)
    return Interpreter(context).run(qasm)


@pytest.mark.parametrize(
    "value, var_type, expected",
    [
        (True, BoolType(), True),
        (np.bool_(True), BoolType(), True),
        (7, IntType(), 7),
        (np.int64(3), UintType(), 3),
        (0.25, FloatType(), 0.25),
        (np.float64(2.5), FloatType(), 2.5),
        (2, FloatType(), 2.0),
        (1, BitType(), 1),
    ],
)
def test_convert_raw_values(value, var_type, expected):
    result = convert_output_value(value, var_type)
    assert result == expected
    assert type(result) is type(expected)


def test_convert_bit_register_from_raw_list_and_tuple():
    bit3 = BitType(size=IntegerLiteral(3))
    assert convert_output_value([0, None, True], bit3) == [0, None, 1]
    assert convert_output_value((1, 0, 1), bit3) == [1, 0, 1]


def test_convert_unconvertible_scalar_raises():
    with pytest.raises(TypeError, match="Cannot convert value 'abc' of output variable 'x'"):
        convert_output_value("abc", IntType(), "x")


def test_convert_unsupported_scalar_type_raises():
    with pytest.raises(TypeError, match="unsupported type AngleType"):
        convert_output_value(IntegerLiteral(1), AngleType(), "theta")


def test_convert_unconvertible_list_raises():
    with pytest.raises(TypeError, match="to a list"):
        convert_output_value(5, BitType(size=IntegerLiteral(2)), "c")


def test_single_path_outputs_follow_measurement_rows():
    context = run_program(
        """
        OPENQASM 3.0;
        output bit[2] c;
        output int ans;
        qubit[2] q;
        c = measure q;
        ans = 42;
        """
    )
    outputs = compute_outputs_single_path(context, [["0", "1"], ["1", "0"]])
    assert outputs == [
        {"c": [0, 1], "ans": 42},
        {"c": [1, 0], "ans": 42},
    ]


def test_single_path_classical_overwrite_after_measurement_wins():
    """``c[0] = 0`` after ``c = measure q`` keeps the measurement in the circuit,
    but the assigned value is reported rather than the sampled bit."""
    context = run_program(
        """
        OPENQASM 3.0;
        output bit[2] c;
        qubit[2] q;
        x q;
        c = measure q;
        c[0] = 0;
        """
    )
    assert context.circuit.measured_qubits == [0, 1]
    assert compute_outputs_single_path(context, [["1", "1"]]) == [{"c": [0, 1]}]


def test_single_path_remeasure_after_overwrite_reads_new_measurement():
    """A measurement into an overwritten element after the write is reported."""
    context = run_program(
        """
        OPENQASM 3.0;
        output bit c;
        qubit[2] q;
        x q;
        c = measure q[0];
        c = 0;
        c = measure q[1];
        """,
        shots=10,
    )
    assert not context.is_branched
    assert context.circuit.measured_qubits == [1]
    assert compute_outputs_single_path(context, [["1"]]) == [{"c": 1}]


def test_single_path_unmeasured_output_register_is_none():
    context = run_program(
        """
        OPENQASM 3.0;
        output bit[2] c;
        output bit b;
        qubit q;
        c[0] = measure q;
        """
    )
    assert compute_outputs_single_path(context, [["1"]]) == [{"c": [1, None], "b": None}]


def test_single_path_output_resolves_global_register_not_shadowing_one():
    """A block-scoped ``bit b`` that shadows ``output bit b`` is a different
    register. The output reports the global one, which was never measured."""
    context = run_program(
        """
        OPENQASM 3.0;
        output bit b;
        qubit q;
        for int i in [0:0] { bit b; b = measure q; }
        """
    )
    (inner,) = context.circuit.measured_registers
    assert inner is not context.register_table.get_register("b")
    assert compute_outputs_single_path(context, [["1"]]) == [{"b": None}]


def test_branched_classical_overwrite_on_taken_path():
    context = run_program(
        """
        OPENQASM 3.0;
        output bit b;
        qubit q;
        h q;
        b = measure q;
        if (b) { b = 0; }
        """,
        shots=50,
    )
    assert context.is_branched
    assert all(o == {"b": 0} for o in compute_outputs_branched(context))


def _run(source, shots):
    return StateVectorSimulator().run(OpenQASMProgram(source=source), shots=shots)


def _assert_outputs_only(result):
    """A program with outputs reports them in place of the measurements."""
    assert result.measurements is None
    assert result.measuredQubits is None


def test_simulator_single_path_outputs_replace_measurements():
    result = _run(
        """
        OPENQASM 3.0;
        output bit[2] c;
        output int ans;
        output bit unset;
        qubit[2] q;
        x q[0];
        c = measure q;
        ans = 42;
        """,
        shots=4,
    )
    _assert_outputs_only(result)
    assert result.outputs == [{"c": [1, 0], "ans": 42, "unset": None}] * 4


def test_simulator_branched_outputs_replace_measurements():
    result = _run(
        """
        OPENQASM 3.0;
        output float f;
        output bit m;
        output int count;
        qubit[2] q;
        f = 2.5;
        h q[0];
        m = measure q[0];
        if (m) { count = 1; x q[1]; } else { count = 2; }
        """,
        shots=20,
    )
    _assert_outputs_only(result)
    assert len(result.outputs) == 20
    assert {output["m"] for output in result.outputs} == {0, 1}
    for output in result.outputs:
        assert output["f"] == 2.5
        # count is set on the branch selected by the measured value of m
        assert output["count"] == (1 if output["m"] else 2)


def test_simulator_branched_whole_register_reported_per_path():
    """A multi-element destination in branched mode populates every element per path."""
    result = _run(
        """
        OPENQASM 3.0;
        output bit[3] c;
        qubit[2] q;
        h q[0];
        c[0:1] = measure q[0:1];
        if (c[0]) { x q[1]; }
        c[2] = measure q[1];
        """,
        shots=40,
    )
    _assert_outputs_only(result)
    assert len(result.outputs) == 40
    for output in result.outputs:
        assert output["c"][0] in (0, 1)
        # q[1] measured |0> mid-circuit on every path
        assert output["c"][1] == 0
        # then flipped exactly on the paths where c[0] measured 1
        assert output["c"][2] == output["c"][0]
        assert all(type(bit) is int for bit in output["c"])


def test_simulator_single_path_reports_each_register():
    result = _run(
        """
        OPENQASM 3.0;
        output bit[2] c;
        output bit d;
        qubit[3] q;
        x q[0];
        x q[2];
        d = measure q[2];
        c = measure q[0:1];
        """,
        shots=5,
    )
    _assert_outputs_only(result)
    assert result.outputs == [{"c": [1, 0], "d": 1}] * 5


def test_simulator_branched_reports_each_register():
    result = _run(
        """
        OPENQASM 3.0;
        output bit m;
        output bit[2] r;
        qubit[3] q;
        h q[0];
        m = measure q[0];
        if (m) { x q[1]; }
        r = measure q[1:2];
        """,
        shots=40,
    )
    _assert_outputs_only(result)
    assert {output["m"] for output in result.outputs} == {0, 1}
    for output in result.outputs:
        assert output["r"] == [output["m"], 0]


def test_simulator_one_output_allows_other_measured_registers():
    """Declaring any ``output`` allows measuring into registers that are not
    outputs and without a destination. Only the outputs are reported."""
    result = _run(
        """
        OPENQASM 3.0;
        output bit c;
        bit d;
        qubit[3] q;
        x q[0];
        c = measure q[0];
        d = measure q[1];
        measure q[2];
        """,
        shots=5,
    )
    _assert_outputs_only(result)
    assert result.outputs == [{"c": 1}] * 5


def test_simulator_shots_zero_with_outputs_gives_no_outputs():
    result = _run(
        "OPENQASM 3.0; output int x; x = 3; qubit q; #pragma braket result probability",
        shots=0,
    )
    assert result.outputs is None


def test_simulator_without_outputs_reports_measurements():
    result = _run("OPENQASM 3.0; bit[2] b; qubit[2] q; x q[1]; b = measure q;", shots=5)
    assert result.outputs is None
    assert result.measuredQubits == [0, 1]
    assert result.measurements == [["0", "1"]] * 5


@pytest.mark.parametrize(
    "source",
    [
        "OPENQASM 3.0; bit c; qubit[2] q; x q; c = measure q[0]; measure q[1];",
        """
        OPENQASM 3.0;
        bit c;
        qubit[2] q;
        h q[0];
        c = measure q[0];
        if (c) { x q[1]; }
        x q[1];
        measure q[1];
        """,
    ],
    ids=["single_path", "branched"],
)
def test_simulator_without_outputs_ignores_measurements_without_destination(source):
    """With a named register measured, ``measure q[1];`` is not reported."""
    result = _run(source, shots=20)
    assert result.outputs is None
    assert result.measuredQubits == [0]
    assert all(len(row) == 1 for row in result.measurements)


def test_simulator_without_outputs_reports_measurements_without_destination():
    result = _run("OPENQASM 3.0; qubit[2] q; x q[0]; measure q;", shots=5)
    assert result.measuredQubits == [0, 1]
    assert result.measurements == [["1", "0"]] * 5


@pytest.mark.parametrize(
    "source",
    [
        "OPENQASM 3.0; bit c; bit d; qubit[2] q; c = measure q[0]; d = measure q[1];",
        """
        OPENQASM 3.0;
        bit c;
        bit d;
        qubit[2] q;
        h q[0];
        c = measure q[0];
        if (c) { x q[1]; }
        d = measure q[1];
        """,
    ],
    ids=["single_path", "branched"],
)
def test_simulator_multiple_registers_without_outputs_raises(source):
    with pytest.raises(ValueError, match="2 classical registers: `c`, `d`.*`output`"):
        _run(source, shots=10)


class TestRegisterMeasurement:
    def test_indexed_measure_into_bit_register(self):
        shots = 8
        qasm = """
        OPENQASM 3.0;
        output bit[2] out;
        qubit[2] q;
        bit[2] c = "00";
        x q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        out = c;
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert result.outputs == [{"out": [0, 1]}] * shots
        for row, output in zip(result.measurements, result.outputs):
            assert output["out"] == [int(bit) for bit in row]

    def test_indexed_measure_into_output_register(self):
        shots = 8
        qasm = """
        OPENQASM 3.0;
        output bit[2] out;
        qubit[2] q;
        x q[1];
        out[0] = measure q[0];
        out[1] = measure q[1];
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert result.outputs == [{"out": [0, 1]}] * shots
        for row, output in zip(result.measurements, result.outputs):
            assert output["out"] == [int(bit) for bit in row]

    def test_whole_register_measure_in_declaration(self):
        shots = 8
        qasm = """
        OPENQASM 3.0;
        output bit[2] out;
        qubit[2] q;
        x q[0];
        bit[2] c = measure q;
        out = c;
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert result.outputs == [{"out": [1, 0]}] * shots
        for row, output in zip(result.measurements, result.outputs):
            assert output["out"] == [int(bit) for bit in row]

    def test_whole_register_measure_into_output_register(self):
        shots = 40
        qasm = """
        OPENQASM 3.0;
        output bit[2] out;
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        out = measure q;
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert len(result.outputs) == shots
        observed = Counter(tuple(output["out"]) for output in result.outputs)
        assert set(observed) <= {(0, 0), (1, 1)}
        for row, output in zip(result.measurements, result.outputs):
            assert output["out"] == [int(bit) for bit in row]

    def test_measurement_read_in_loop_scope(self):
        shots = 8
        qasm = """
        OPENQASM 3.0;
        output bit[2] out;
        out = "00";
        qubit[1] q;
        bit c = 0;
        for int i in [0:1] {
            x q[0];
            bit m;
            m = measure q[0];
            c = m;
            out[i] = c;
            reset q[0];
        }
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert result.outputs == [{"out": [1, 1]}] * shots

    def test_indexed_measure_into_int_array(self):
        shots = 4
        qasm = """
        OPENQASM 3.0;
        output array[int[8], 2] out;
        qubit[2] q;
        array[int[8], 2] a = {0, 0};
        x q[0];
        a[0] = measure q[0];
        a[1] = measure q[1];
        out = a;
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert result.outputs == [{"out": [1, 0]}] * shots

    def test_whole_register_measure_into_int_array(self):
        shots = 4
        qasm = """
        OPENQASM 3.0;
        output array[int[8], 2] out;
        qubit[2] q;
        x q[1];
        array[int[8], 2] a = measure q;
        out = a;
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=shots)
        assert result.outputs == [{"out": [0, 1]}] * shots

    def test_bit_register_size_mismatch(self):
        qasm = """
        OPENQASM 3.0;
        output bit[3] out;
        qubit[2] q;
        out = measure q;
        """
        with pytest.raises(
            ValueError, match=r"\(2\) does not match size of classical register 'out' \(3\)"
        ):
            StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=2)

    def test_int_array_size_mismatch(self):
        qasm = """
        OPENQASM 3.0;
        output array[int[8], 3] out;
        qubit[2] q;
        out = measure q;
        """
        with pytest.raises(
            ValueError, match=r"\(2\) does not match size of classical register 'out' \(3\)"
        ):
            StateVectorSimulator().run(OpenQASMProgram(source=qasm), shots=2)
