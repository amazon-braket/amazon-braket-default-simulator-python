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

from braket.default_simulator import gate_operations
from braket.default_simulator.openqasm.circuit import Circuit
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    AngleType,
    ArrayType,
    BitType,
    BooleanLiteral,
    BoolType,
    FloatLiteral,
    FloatType,
    Identifier,
    IntegerLiteral,
    IntType,
)
from braket.default_simulator.openqasm.program_context import ProgramContext, ScopedTable

boolean = BoolType()
int_8 = IntType(IntegerLiteral(8))
int_16 = IntType(IntegerLiteral(16))
float_8 = FloatType(IntegerLiteral(8))
float_16 = FloatType(IntegerLiteral(16))


def test_variable_declaration():
    context = ProgramContext()
    context.declare_variable("x", int_8, IntegerLiteral(10), True)
    context.declare_variable("y", float_16, FloatLiteral(1.34), False)
    context.declare_variable("z", boolean, BooleanLiteral(False), False)

    def assert_scope_0():
        assert context.get_type("x") == int_8
        assert context.get_type("y") == float_16
        assert context.get_type("z") == boolean

        assert context.get_const("x")
        assert not context.get_const("y")
        assert not context.get_const("z")

        assert context.get_value("x") == IntegerLiteral(10)
        assert context.get_value("y") == FloatLiteral(1.34)
        assert context.get_value("z") == BooleanLiteral(False)

        with pytest.raises(KeyError):
            context.get_type("a")

        with pytest.raises(KeyError):
            context.get_value("a")

    assert_scope_0()

    with context.enter_scope():
        context.declare_variable("x", int_16, IntegerLiteral(20), False)
        context.declare_variable("y", float_8, FloatLiteral(2.68), True)
        context.declare_variable("a", boolean, BooleanLiteral(True), False)

        assert context.get_type("x") == int_16
        assert context.get_type("y") == float_8
        assert context.get_type("z") == boolean
        assert context.get_type("a") == boolean

        assert not context.get_const("x")
        assert context.get_const("y")
        assert not context.get_const("z")
        assert not context.get_const("a")

        assert context.get_value("x") == IntegerLiteral(20)
        assert context.get_value("y") == FloatLiteral(2.68)
        assert context.get_value("z") == BooleanLiteral(False)
        assert context.get_value("a") == BooleanLiteral(True)

    assert_scope_0()


def test_repr():
    context = ProgramContext()
    context.declare_variable("x", int_8, IntegerLiteral(10), True)
    context.declare_variable("y", float_16, FloatLiteral(1.34), False)
    context.declare_variable("z", boolean, BooleanLiteral(False), False)

    context.add_qubits("q")

    with context.enter_scope():
        context.declare_variable("x", int_16, IntegerLiteral(20), False)
        context.declare_variable("y", float_8, FloatLiteral(2.68), True)
        context.declare_variable("a", boolean, BooleanLiteral(True), False)

        assert repr(context) == (
            """Symbols
SCOPE LEVEL 0
x	Symbol<IntType(span=None, size=IntegerLiteral(span=None, value=8)), const=True>
y	Symbol<FloatType(span=None, size=IntegerLiteral(span=None, value=16)), const=False>
z	Symbol<BoolType(span=None), const=False>
q	Symbol<<class 'braket.default_simulator.openqasm.parser.openqasm_ast.Identifier'>, const=False>
SCOPE LEVEL 1
x	Symbol<IntType(span=None, size=IntegerLiteral(span=None, value=16)), const=False>
y	Symbol<FloatType(span=None, size=IntegerLiteral(span=None, value=8)), const=True>
a	Symbol<BoolType(span=None), const=False>

Data
SCOPE LEVEL 0
x	IntegerLiteral(span=None, value=10)
y	FloatLiteral(span=None, value=1.34)
z	BooleanLiteral(span=None, value=False)
q	Identifier(span=None, name='q')
SCOPE LEVEL 1
x	IntegerLiteral(span=None, value=20)
y	FloatLiteral(span=None, value=2.68)
a	BooleanLiteral(span=None, value=True)

Gates
SCOPE LEVEL 0
SCOPE LEVEL 1

Qubits
q	(0,)"""
        )


def test_delete_from_scope():
    table = ScopedTable("title")
    table["x"] = 1
    table.push_scope()
    assert table._scopes == [{"x": 1}, {}]
    del table["x"]
    assert table._scopes == [{}, {}]

    undefined_key = "Undefined key: x"
    with pytest.raises(KeyError, match=undefined_key):
        del table["x"]


def test_prebuilt_circuit():
    circuit = Circuit()
    circuit.add_instruction(gate_operations.Hadamard([0]))
    context = ProgramContext(circuit)
    context.add_gate_instruction("cnot", (0, 1), [], ctrl_modifiers=[], power=1)
    assert context.circuit.instructions == [
        gate_operations.Hadamard([0]),
        gate_operations.CX([0, 1]),
    ]


def test_add_barrier_method_exists():
    """Test that add_barrier method exists and can be called without errors."""
    context = ProgramContext()

    # Should not raise any exceptions
    context.add_barrier([0, 1])  # With specific qubits
    context.add_barrier(None)  # Global barrier
    context.add_barrier([])  # Empty qubit list


def test_add_barrier_is_noop():
    """Test that add_barrier doesn't add any instructions to the circuit."""
    context = ProgramContext()
    initial_instruction_count = len(context.circuit.instructions)

    # Add barriers with different parameters
    context.add_barrier([0, 1])
    context.add_barrier(None)
    context.add_barrier([2])

    # Circuit should remain unchanged
    assert len(context.circuit.instructions) == initial_instruction_count


def run_output_program(qasm):
    """Interpret a program and flush pending measurements into the circuit."""
    context = Interpreter().run(qasm)
    _ = context.circuit  # trigger pending-MCM flush (records output bindings)
    return context


def test_unsupported_output_type():
    with pytest.raises(NotImplementedError, match="type AngleType are not supported"):
        Interpreter().run("OPENQASM 3.0;\noutput angle[4] theta;")


def test_unsupported_output_array_base_type():
    context = ProgramContext()
    with pytest.raises(NotImplementedError, match="type AngleType are not supported"):
        context.add_output_declaration("a", ArrayType(AngleType(), [IntegerLiteral(2)]))


def test_output_binding_whole_register():
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit[2] c;
        qubit[2] q;
        c = measure q;
        """
    )
    assert context.output_bindings == {"c": {0: 0, 1: 1}}


def test_output_binding_scalar_bit():
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit b;
        qubit q;
        b = measure q;
        """
    )
    assert context.output_bindings == {"b": {None: 0}}


def test_output_binding_indexed_destination():
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit[2] c;
        qubit[2] q;
        c[1] = measure q[0];
        """
    )
    assert context.output_bindings == {"c": {1: 1}}


def test_output_binding_non_output_destination_ignored():
    context = run_output_program(
        """
        OPENQASM 3.0;
        output int x;
        bit[2] c;
        qubit[2] q;
        c = measure q;
        """
    )
    assert context.output_bindings == {}


def test_record_output_binding_early_returns():
    context = ProgramContext()
    context._record_output_binding(None, [None], [0])
    context._record_output_binding(Identifier(name="b"), [], [])
    assert context.output_bindings == {}


def test_element_index_of_one_register_does_not_claim_another_variables_bit():
    """``c2[0]`` is element 0 of c2, not global classical bit 0, so it must not
    displace the bit already assigned to c1."""
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit c1;
        output bit[2] c2;
        qubit[2] q;
        c1 = measure q[0];
        c2[0] = measure q[1];
        """
    )
    assert context.circuit.measured_qubits == [0, 1]
    assert context.circuit.target_classical_indices == [0, 1]
    assert context.output_bindings == {"c1": {None: 0}, "c2": {0: 1}}


def test_same_element_index_in_two_registers_gets_distinct_bits():
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit[2] a;
        output bit[2] b;
        qubit[2] q;
        a[0] = measure q[0];
        b[0] = measure q[1];
        """
    )
    assert context.circuit.measured_qubits == [0, 1]
    assert context.output_bindings == {"a": {0: 0}, "b": {0: 1}}


def test_interleaved_registers_keep_their_own_elements():
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit[2] a;
        output bit[2] b;
        qubit[4] q;
        a[0] = measure q[0];
        b[0] = measure q[1];
        a[1] = measure q[2];
        b[1] = measure q[3];
        """
    )
    assert context.output_bindings == {"a": {0: 0, 1: 2}, "b": {0: 1, 1: 3}}


def test_remeasuring_an_element_reuses_its_classical_bit():
    """Remeasuring the same destination overwrites it rather than consuming a
    second classical bit."""
    context = run_output_program(
        """
        OPENQASM 3.0;
        output bit c;
        qubit[2] q;
        c = measure q[0];
        c = measure q[1];
        """
    )
    assert context.circuit.target_classical_indices == [0]
    assert context.circuit.measured_qubits == [1]
    assert context.output_bindings == {"c": {None: 0}}


def test_resolve_classical_indices_without_destination_passes_through():
    context = ProgramContext()
    assert context._resolve_classical_indices(None, [3, 4], (0, 1), [None, None]) == [3, 4]
    assert context._resolve_classical_indices(None, None, (0,), [None]) is None
    assert context._destination_elements(None, None, (0, 1)) == [None, None]


def test_declared_type_of_unknown_name_is_none():
    assert ProgramContext()._declared_type("nope") is None


def test_register_size_of_array_type():
    from braket.default_simulator.openqasm.program_context import _register_size

    assert _register_size(ArrayType(IntType(None), [IntegerLiteral(4)])) == 4
    assert _register_size(BitType(IntegerLiteral(2))) == 2
    assert _register_size(BitType(None)) is None
    assert _register_size(None) is None


def test_update_identifier_target_materializes_missing_register_value():
    """A register-typed variable whose value is not yet an array is
    materialized before its elements are written."""
    from braket.default_simulator.openqasm.simulation_path import FramedVariable, SimulationPath

    context = ProgramContext()
    path = SimulationPath()
    path.set_variable(
        "c",
        FramedVariable(
            name="c",
            var_type=BitType(IntegerLiteral(2)),
            value=None,  # not an ArrayLiteral yet
            is_const=False,
            frame_number=path.frame_number,
        ),
    )
    path._measurements = {0: [1], 1: [0]}
    context._update_identifier_target(path, (0, 1), Identifier(name="c"))
    assert [element.value for element in path.get_variable("c").value.values] == [1, 0]
