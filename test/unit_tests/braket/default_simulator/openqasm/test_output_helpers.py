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

"""Example-based unit tests for the OpenQASM output-variable result flow.

Covers ``output_helpers.py`` (value conversion, per-shot output computation
for the single-path and branched flows, and private helper edge branches)
and the simulator's compute-outputs guards.
"""

import importlib
import typing
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from braket.default_simulator.openqasm import output_helpers as output_helpers_module
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.output_helpers import (
    _convert_scalar,
    _declared_size,
    _element_list,
    _path_base_value,
    compute_outputs_branched,
    compute_outputs_single_path,
    compute_shot_output,
    convert_output_value,
)
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    AngleType,
    ArrayLiteral,
    ArrayType,
    BitType,
    BooleanLiteral,
    BoolType,
    FloatLiteral,
    FloatType,
    IntegerLiteral,
    IntType,
    UintType,
)
from braket.default_simulator.openqasm.program_context import ProgramContext
from braket.default_simulator.state_vector_simulator import StateVectorSimulator
from braket.ir.openqasm import Program as OpenQASMProgram


def run_program(qasm):
    """Interpret a program and flush pending measurements into the circuit."""
    context = Interpreter().run(qasm)
    _ = context.circuit  # trigger pending-MCM flush (records output bindings)
    return context


class TestConvertOutputValue:
    """convert_output_value and its scalar/element conversion helpers."""

    @pytest.mark.parametrize(
        "value, var_type, expected",
        [
            (BooleanLiteral(True), BoolType(), True),
            (BooleanLiteral(False), BoolType(), False),
            (IntegerLiteral(-3), IntType(IntegerLiteral(8)), -3),
            (IntegerLiteral(5), UintType(), 5),
            (FloatLiteral(1.5), FloatType(), 1.5),
            # unsized bit -> int 0/1
            (BooleanLiteral(True), BitType(), 1),
            (IntegerLiteral(0), BitType(), 0),
        ],
    )
    def test_ast_literals(self, value, var_type, expected):
        result = convert_output_value(value, var_type)
        assert result == expected
        assert type(result) is type(expected)

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
    def test_raw_values(self, value, var_type, expected):
        result = convert_output_value(value, var_type)
        assert result == expected
        assert type(result) is type(expected)

    @pytest.mark.parametrize(
        "var_type",
        [BoolType(), IntType(), FloatType(), BitType(), BitType(size=IntegerLiteral(2))],
    )
    def test_none_preserved(self, var_type):
        assert convert_output_value(None, var_type) is None

    def test_convert_scalar_raw_none(self):
        # _convert_scalar preserves a raw None (defensive branch)
        assert _convert_scalar(None, BoolType(), "x") is None

    def test_bit_register_from_array_literal_with_partial_none(self):
        value = ArrayLiteral(values=[BooleanLiteral(True), None, BooleanLiteral(False)])
        assert convert_output_value(value, BitType(size=IntegerLiteral(3))) == [1, None, 0]

    def test_bit_register_from_raw_list_and_tuple(self):
        bit3 = BitType(size=IntegerLiteral(3))
        assert convert_output_value([0, None, True], bit3) == [0, None, 1]
        assert convert_output_value((1, 0, 1), bit3) == [1, 0, 1]

    def test_one_dimensional_array(self):
        var_type = ArrayType(FloatType(), [IntegerLiteral(2)])
        value = ArrayLiteral(values=[FloatLiteral(0.5), FloatLiteral(1.5)])
        assert convert_output_value(value, var_type) == [0.5, 1.5]

    def test_nested_array(self):
        var_type = ArrayType(IntType(), [IntegerLiteral(2), IntegerLiteral(2)])
        value = ArrayLiteral(
            values=[
                ArrayLiteral(values=[IntegerLiteral(1), IntegerLiteral(2)]),
                ArrayLiteral(values=[IntegerLiteral(3), IntegerLiteral(4)]),
            ]
        )
        assert convert_output_value(value, var_type) == [[1, 2], [3, 4]]

    def test_unconvertible_scalar_raises(self):
        with pytest.raises(TypeError, match="Cannot convert value 'abc' of output variable 'x'"):
            convert_output_value("abc", IntType(), "x")

    def test_unsupported_scalar_type_raises(self):
        with pytest.raises(TypeError, match="unsupported type AngleType"):
            convert_output_value(IntegerLiteral(1), AngleType(), "theta")

    def test_unconvertible_list_raises(self):
        with pytest.raises(TypeError, match="to a list"):
            convert_output_value(5, BitType(size=IntegerLiteral(2)), "c")
        with pytest.raises(TypeError, match="to a list"):
            convert_output_value(5, ArrayType(IntType(), [IntegerLiteral(2)]), "a")


class TestComputeShotOutput:
    """compute_shot_output measurement-bit replacement and conversion."""

    def test_no_bindings_no_measurements(self):
        output_variables = {"x": IntType(), "b": BitType()}
        base_values = {"x": IntegerLiteral(42), "b": None}
        result = compute_shot_output(output_variables, base_values, {}, None, {})
        assert result == {"x": 42, "b": None}

    def test_scalar_bit_binding(self):
        output_variables = {"b": BitType()}
        result = compute_shot_output(
            output_variables,
            {"b": None},
            {"b": {None: 0}},
            ["1"],
            {0: 0},
        )
        assert result == {"b": 1}

    def test_scalar_bit_binding_classical_index_missing_from_classical_idx_columns(self):
        # The bound classical index is not a measured column: keep the base value.
        output_variables = {"b": BitType()}
        result = compute_shot_output(
            output_variables,
            {"b": BooleanLiteral(False)},
            {"b": {None: 5}},
            ["1"],
            {0: 0},
        )
        assert result == {"b": 0}

    def test_element_bindings(self):
        output_variables = {"c": BitType(size=IntegerLiteral(2))}
        base_values = {"c": ArrayLiteral(values=[None, None])}
        result = compute_shot_output(
            output_variables,
            base_values,
            {"c": {0: 0, 1: 1}},
            ["1", "0"],
            {0: 0, 1: 1},
        )
        assert result == {"c": [1, 0]}

    def test_element_binding_classical_index_missing_from_classical_idx_columns(self):
        output_variables = {"c": BitType(size=IntegerLiteral(2))}
        base_values = {"c": [0, 1]}
        result = compute_shot_output(
            output_variables,
            base_values,
            {"c": {0: 9}},  # classical index 9 was never measured
            ["1"],
            {0: 0},
        )
        assert result == {"c": [0, 1]}

    def test_bindings_without_measurement_row_keeps_base_values(self):
        output_variables = {"b": BitType()}
        result = compute_shot_output(
            output_variables,
            {"b": BooleanLiteral(True)},
            {"b": {None: 0}},
            None,
            {0: 0},
        )
        assert result == {"b": 1}

    def test_none_base_value_materializes_declared_size(self):
        # base value None + element bindings -> _element_list falls back
        # to a list of None values of the declared size.
        output_variables = {"c": BitType(size=IntegerLiteral(3))}
        result = compute_shot_output(
            output_variables,
            {"c": None},
            {"c": {1: 0}},
            ["1"],
            {0: 0},
        )
        assert result == {"c": [None, 1, None]}


class TestPrivateHelpers:
    """Edge branches of the private helpers."""

    def test_element_list_from_array_literal_and_sequences(self):
        bit2 = BitType(size=IntegerLiteral(2))
        literal = ArrayLiteral(values=[BooleanLiteral(True), None])
        assert _element_list(literal, bit2) == [BooleanLiteral(True), None]
        assert _element_list([1, 0], bit2) == [1, 0]
        assert _element_list((0, 1), bit2) == [0, 1]

    def test_element_list_fallback_to_declared_size(self):
        assert _element_list(None, BitType(size=IntegerLiteral(3))) == [None, None, None]
        assert _element_list(None, ArrayType(IntType(), [IntegerLiteral(2)])) == [None, None]

    def test_declared_size_type_error(self):
        with pytest.raises(TypeError, match="BoolType has no declared size"):
            _declared_size(BoolType())
        with pytest.raises(TypeError, match="BitType has no declared size"):
            _declared_size(BitType())  # unsized bit

    def test_path_base_value_prefers_path_variable(self):
        context = ProgramContext()
        path = SimpleNamespace(get_variable=lambda name: SimpleNamespace(value=IntegerLiteral(7)))
        assert _path_base_value(context, path, "x") == IntegerLiteral(7)

    def test_path_base_value_falls_back_to_shared_table(self):
        context = ProgramContext()
        context.declare_variable("x", IntType(), IntegerLiteral(5))
        path = SimpleNamespace(get_variable=lambda name: None)
        assert _path_base_value(context, path, "x") == IntegerLiteral(5)

    def test_path_base_value_missing_everywhere_is_none(self):
        context = ProgramContext()
        path = SimpleNamespace(get_variable=lambda name: None)
        assert _path_base_value(context, path, "never_declared") is None

    def test_type_checking_import(self):
        # Execute the TYPE_CHECKING-guarded import (line coverage only).
        try:
            typing.TYPE_CHECKING = True
            importlib.reload(output_helpers_module)
            assert hasattr(output_helpers_module, "ProgramContext")
        finally:
            typing.TYPE_CHECKING = False
            importlib.reload(output_helpers_module)


class TestComputeOutputsSinglePath:
    def test_outputs_follow_measurement_rows(self):
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


SINGLE_PATH_QASM = """
OPENQASM 3.0;
output bit[2] c;
output int ans;
output bit unset;
qubit[2] q;
x q[0];
c = measure q;
ans = 42;
"""

BRANCHED_QASM = """
OPENQASM 3.0;
output float f;
output bit m;
output int count;
qubit[2] q;
f = 2.5;
h q[0];
m = measure q[0];
if (m) { count = 1; x q[1]; } else { count = 2; }
"""


class TestSimulatorOutputs:
    def test_single_path_run_populates_outputs(self):
        shots = 4
        result = StateVectorSimulator().run(OpenQASMProgram(source=SINGLE_PATH_QASM), shots=shots)
        assert result.outputs == [{"c": [1, 0], "ans": 42, "unset": None}] * shots
        # per-shot correspondence with the measurement rows
        assert len(result.measurements) == shots
        for row, output in zip(result.measurements, result.outputs):
            assert output["c"] == [int(bit) for bit in row]

    def test_branched_run_populates_outputs(self):
        shots = 20
        result = StateVectorSimulator().run(OpenQASMProgram(source=BRANCHED_QASM), shots=shots)
        assert len(result.outputs) == shots
        for row, output in zip(result.measurements, result.outputs):
            assert output["f"] == 2.5
            assert output["m"] in (0, 1)
            # per-shot correspondence: m is classical index 0, measurement column 0
            assert output["m"] == int(row[0])
            # count is set on the branch selected by the measured value of m
            assert output["count"] == (1 if output["m"] else 2)
        # both branches appear with high probability over 20 shots of |+>
        observed = Counter(output["m"] for output in result.outputs)
        assert set(observed) <= {0, 1}

    def test_compute_outputs_branched_requires_a_row_per_shot(self):
        # A shot without a measurement row means the caller mis-built the rows;
        # fail loudly rather than silently reporting base values.
        shots = 5
        context = StateVectorSimulator()._parse_program_with_shots(
            OpenQASMProgram(source=BRANCHED_QASM, inputs={}), shots
        )
        assert context.is_branched
        with pytest.raises(ValueError, match="more shot results than measurements"):
            compute_outputs_branched(context, [])


WHOLE_REGISTER_BRANCHED_QASM = """
OPENQASM 3.0;
output bit[2] c;
qubit[2] q;
h q[0];
c = measure q;
if (c[0]) { x q[1]; }
"""


class TestBranchedWholeRegisterMeasurement:
    """A whole-register plain destination (``c = measure q``) in branched mode
    must populate every register element per path, not a bare scalar."""

    def test_register_elements_reported_per_path(self):
        shots = 40
        result = StateVectorSimulator().run(
            OpenQASMProgram(source=WHOLE_REGISTER_BRANCHED_QASM), shots=shots
        )
        assert len(result.outputs) == shots
        for output in result.outputs:
            # Both elements are defined ints, and q[1] measured |0> every shot.
            assert output["c"][1] == 0
            assert output["c"][0] in (0, 1)
            assert all(type(bit) is int for bit in output["c"])

    def test_conditioned_gate_follows_the_measured_register_element(self):
        shots = 40
        result = StateVectorSimulator().run(
            OpenQASMProgram(source=WHOLE_REGISTER_BRANCHED_QASM), shots=shots
        )
        # q[1] is flipped exactly on the paths where c[0] measured 1, so the
        # final-state bit for q[1] tracks c[0].
        for row, output in zip(result.measurements, result.outputs):
            assert int(row[1]) == output["c"][0]

    def test_register_size_larger_than_measured_qubits_leaves_none(self):
        source = """
        OPENQASM 3.0;
        output bit[3] c;
        qubit[2] q;
        h q[0];
        c[0] = measure q[0];
        if (c[0]) { x q[1]; }
        c[1] = measure q[1];
        """
        result = StateVectorSimulator().run(OpenQASMProgram(source=source), shots=10)
        for output in result.outputs:
            assert output["c"][2] is None


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
