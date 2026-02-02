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
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    BooleanLiteral,
    BoolType,
    FloatLiteral,
    FloatType,
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
