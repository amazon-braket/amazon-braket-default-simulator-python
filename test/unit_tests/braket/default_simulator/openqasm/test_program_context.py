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
    ArrayLiteral,
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
from braket.default_simulator.state_vector_simulator import StateVectorSimulator

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


def test_bit_declarations_create_registers():
    context = ProgramContext()
    context.declare_variable("b", BitType(size=None), None)
    context.declare_variable("c", BitType(IntegerLiteral(3)), ArrayLiteral([None] * 3))
    context.declare_variable("x", int_8, IntegerLiteral(0))
    b, c = context.circuit.classical_registers
    assert (b.name, b.size) == ("b", 1)
    assert (c.name, c.size) == ("c", 3)
    assert context.register_table.get_register("b") is b
    assert context.register_table.get_register("c") is c
    assert context.register_table.get_register("x") is None
    assert context.register_table.get_register("undeclared") is None


def test_unevaluated_size_falls_back_to_value_width():
    """Subroutine parameters keep their unevaluated type. The size comes from the value."""
    context = ProgramContext()
    context.declare_variable(
        "p", BitType(Identifier("n")), ArrayLiteral([BooleanLiteral(False)] * 4)
    )
    context.declare_variable("s", BitType(Identifier("n")), BooleanLiteral(True))
    p, s = context.circuit.classical_registers
    assert p.size == 4
    assert s.size == 1


def test_repr_includes_register_table():
    context = ProgramContext()
    context.declare_variable("b", BitType(size=None), None)
    assert "Registers" in repr(context.register_table)


def _branched_context(qasm, shots=100):
    context = StateVectorSimulator().create_program_context()
    context._shots = shots
    return Interpreter(context).run(source=qasm)


def _path_bit(path, name):
    value = path.get_variable(name).value
    return int(getattr(value, "value", value))


def test_mcm_dependent_declaration_has_one_register_and_a_value_per_path():
    """``bit c = b;`` after branching is one declaration with a value on each path."""
    context = _branched_context(
        "qubit[2] q; bit b; h q[0]; b = measure q[0]; if (b) { x q[1]; } bit c = b;"
    )
    assert len(context.active_paths) == 2
    assert [r.name for r in context.circuit.classical_registers] == ["b", "c"]
    c = context.circuit.classical_registers[1]
    assert context.register_table.get_register("c") is c

    for path in context.active_paths:
        assert _path_bit(path, "c") == _path_bit(path, "b")
    assert {_path_bit(path, "c") for path in context.active_paths} == {0, 1}


def test_mcm_dependent_declaration_initializer_sees_outer_variable():
    """The initializer is evaluated before the declaration, so ``bit b = b;`` in an
    inner scope reads the outer ``b`` on each path."""
    context = _branched_context(
        "qubit q; bit b; bit seen; h q; b = measure q; if (b) { x q; } "
        "for int i in [0:0] { bit b = b; seen = b; }"
    )
    assert len(context.active_paths) == 2
    assert [r.name for r in context.circuit.classical_registers] == ["b", "seen", "b"]
    for path in context.active_paths:
        assert _path_bit(path, "seen") == _path_bit(path, "b")
    assert {_path_bit(path, "seen") for path in context.active_paths} == {0, 1}


def test_mcm_dependent_declaration_without_branching():
    """With ``shots == 0`` nothing branches, so the declaration keeps its one value."""
    context = _branched_context(
        "qubit q; bit b; x q; b = measure q; int n = int(b) + 5;",
        shots=0,
    )
    assert not context.is_branched
    assert context.get_value("n") == IntegerLiteral(5)


def test_loop_declarations_still_create_one_register_per_iteration():
    """Paths are alternatives within a shot, so a declaration replayed per path has one
    register. Loop iterations run one after another within a shot, so each iteration
    declares a new variable and gets its own register."""
    context = _branched_context(
        "qubit q; bit b; h q; b = measure q; if (b) { x q; } for int i in [0:2] { bit r; }"
    )
    assert [r.name for r in context.circuit.classical_registers] == ["b", "r", "r", "r"]


def test_shadowing_declaration_resolves_per_scope():
    context = ProgramContext()
    context.declare_variable("b", BitType(size=None), None)
    outer = context.register_table.get_register("b")
    context.push_scope()
    context.declare_variable("b", BitType(size=None), None)
    inner = context.register_table.get_register("b")
    assert inner is not outer
    context.pop_scope()
    assert context.register_table.get_register("b") is outer
    assert context.circuit.classical_registers == [outer, inner]
