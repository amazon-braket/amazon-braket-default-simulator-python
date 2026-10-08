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

from typing import NamedTuple

import pytest

from braket.default_simulator import gate_operations
from braket.default_simulator.openqasm.circuit import Circuit
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    ArrayLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    DiscreteSet,
    FloatLiteral,
    FloatType,
    Identifier,
    IndexedIdentifier,
    IntegerLiteral,
    IntType,
)
from braket.default_simulator.openqasm.program_context import ProgramContext, ScopedTable
from braket.default_simulator.state_vector_simulator import StateVectorSimulator
from braket.ir.openqasm import Program as OpenQASMProgram

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


class Slot(NamedTuple):
    """One measurement slot: which register element reports which qubit."""

    register: str | None
    element: int
    qubit: int


def _slots(circuit):
    return [
        Slot(register.name, element, qubit)
        for register, element, qubit in circuit.measurement_slots
    ]


def _build(qasm, shots=0):
    """Interpret ``qasm`` with a simulator-backed context and return the context."""
    context = StateVectorSimulator().create_program_context()
    context._shots = shots
    return Interpreter(context).run(source=qasm)


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


def _path_bit(path, name):
    value = path.get_variable(name).value
    return int(getattr(value, "value", value))


def test_mcm_dependent_declaration_has_one_register_and_a_value_per_path():
    """``bit c = b;`` after branching is one declaration with a value on each path."""
    context = _build(
        "qubit[2] q; bit b; h q[0]; b = measure q[0]; if (b) { x q[1]; } bit c = b;",
        shots=100,
    )
    assert len(context.active_paths) == 2
    assert [r.name for r in context.circuit.classical_registers] == ["b", "c"]
    c = context.circuit.classical_registers[1]
    assert context.register_table.get_register("c") is c

    for path in context.active_paths:
        assert _path_bit(path, "c") == _path_bit(path, "b")
    assert {_path_bit(path, "c") for path in context.active_paths} == {0, 1}


def test_mcm_dependent_declaration_without_branching():
    """With ``shots == 0`` nothing branches, so the declaration keeps its one value."""
    context = _build(
        "qubit q; bit b; x q; b = measure q; int n = int(b) + 5;",
        shots=0,
    )
    assert not context.is_branched
    assert context.get_value("n") == IntegerLiteral(5)


@pytest.mark.parametrize(
    "destination, expected",
    [
        ("c[1]", [Slot(register="c", element=1, qubit=0)]),
        (
            "c[{2, 0}]",
            [Slot(register="c", element=0, qubit=1), Slot(register="c", element=2, qubit=0)],
        ),
        (
            "c[0:1]",
            [Slot(register="c", element=0, qubit=0), Slot(register="c", element=1, qubit=1)],
        ),
    ],
)
def test_resolve_destination_indexed_forms(destination, expected):
    qubits = "q[0]" if destination == "c[1]" else "q[0:1]"
    circuit = _build(f"bit[3] c; qubit[2] q; {destination} = measure {qubits};").circuit
    assert _slots(circuit) == expected


# A classical write to a measured bit register element releases the measurement.
# The element loses its measurement source and is no longer a measurement slot.
# The qubit stays part of the circuit.


def test_overwrite_releases_pending_scalar_measurement():
    circuit = _build("bit b; qubit q; h q; b = measure q; b = 0;", shots=10).circuit
    (b,) = circuit.classical_registers
    assert b.sources == [None]
    assert circuit.measurement_slots == []
    assert circuit.measured_registers == []
    assert circuit.qubit_set == {0}


def test_element_overwrite_keeps_other_elements():
    qasm = "bit[2] c; qubit[2] q; c = measure q; c[0] = 0;"
    circuit = _build(qasm, shots=10).circuit
    (c,) = circuit.classical_registers
    assert c.sources == [None, 1]
    assert _slots(circuit) == [Slot(register="c", element=1, qubit=1)]
    assert circuit.qubit_set == {0, 1}


@pytest.mark.parametrize(
    "lvalue, expected_sources",
    [
        ('c[1:2] = "00"', [0, None, None]),
        ("c[-1] = 0", [0, 1, None]),
    ],
)
def test_overwrite_indexed_lvalue_forms(lvalue, expected_sources):
    qasm = f"bit[3] c; qubit[3] q; c = measure q; {lvalue};"
    circuit = _build(qasm, shots=10).circuit
    (c,) = circuit.classical_registers
    assert c.sources == expected_sources


def test_overwrite_discrete_set_lvalue_indices():
    lvalue = IndexedIdentifier(
        Identifier("c"), [DiscreteSet([IntegerLiteral(2), IntegerLiteral(0)])]
    )
    assert ProgramContext._indexed_elements(lvalue, 3) == [2, 0]


def test_overwrite_of_measurement_already_in_circuit():
    """With ``shots == 0`` a read forces the pending measurement into the circuit
    before the overwrite, exercising the bound-source path."""
    qasm = "bit b; qubit q; h q; b = measure q; int x = b; b = 1;"
    context = _build(qasm, shots=0)
    (b,) = context.circuit.classical_registers
    assert b.sources == [None]
    assert context.get_value("b") == BooleanLiteral(True)


def test_overwrite_of_unmeasured_bit_is_a_plain_assignment():
    circuit = _build("bit b; qubit q; b = 1; measure q;", shots=10).circuit
    assert _slots(circuit) == [Slot(register=None, element=0, qubit=0)]


def test_branched_overwrite_leaves_circuit_alone():
    """After branching, the per-path outcomes are authoritative. The shared circuit
    keeps the measurement source and the slot reports the measured value."""
    qasm = """
    bit b;
    qubit q;
    h q;
    b = measure q;
    if (b) { b = 0; }
    """
    context = _build(qasm, shots=100)
    assert context.is_branched
    (b,) = context.circuit.classical_registers
    assert b.sources == [0]
    result = StateVectorSimulator().run_openqasm(OpenQASMProgram(source=qasm), shots=200)
    assert result.measuredQubits == [0]
    assert {"".join(m) for m in result.measurements} == {"0", "1"}


# A deferred measurement may be forced while its destination is shadowed, here by a
# subroutine's local ``bit b`` when the subroutine applies a gate to the qubit.
_SHADOWED_DEFERRED_QASM = """
def flip(qubit r) { bit b; x r; }
qubit q;
bit b;
h q;
b = measure q;
flip(q);
"""


def test_shadowed_deferred_measurement_without_shots():
    circuit = _build(_SHADOWED_DEFERRED_QASM, shots=0).circuit
    assert _slots(circuit) == [Slot(register="b", element=0, qubit=0)]


def test_shadowed_deferred_measurement_with_shots():
    context = _build(_SHADOWED_DEFERRED_QASM, shots=50)
    assert context.is_branched
    outer, inner = context.circuit.classical_registers
    assert inner.sources == [None]
    assert all((outer, 0) in path.mcm_outcomes for path in context.active_paths)


# Every pending measurement is applied when one of them is forced, so none is lost.


def test_later_pending_measurement_survives_partial_read():
    """Reading ``b[0]`` branches, and ``b[1]``'s deferred measurement is applied with
    it rather than left pending (previously it silently vanished from the results)."""
    qasm = """
    bit[2] b;
    qubit[2] q;
    h q[0];
    x q[1];
    b[0] = measure q[0];
    b[1] = measure q[1];
    int x = b[0];
    """
    context = _build(qasm, shots=20)
    assert context.is_branched
    assert context._pending_mcm_targets == []
    assert _slots(context.circuit) == [
        Slot(register="b", element=0, qubit=0),
        Slot(register="b", element=1, qubit=1),
    ]
    (b,) = context.circuit.classical_registers
    assert all(path.mcm_outcomes[(b, 1)] == 1 for path in context.active_paths)
    result = StateVectorSimulator().run_openqasm(OpenQASMProgram(source=qasm), shots=20)
    assert result.measuredQubits == [0, 1]
    assert {"".join(m) for m in result.measurements} <= {"01", "11"}


def test_control_flow_without_shots_reads_measurement_as_zero():
    """With ``shots == 0`` a deferred measurement used in control flow is recorded
    and its bit reads as 0, instead of the condition reading an unset variable."""
    qasm = """
    bit b;
    qubit[2] q;
    x q[0];
    b = measure q[0];
    if (b == 1) { x q[1]; }
    """
    context = _build(qasm, shots=0)
    assert not context.is_branched
    assert _slots(context.circuit) == [Slot(register="b", element=0, qubit=0)]
    assert context.get_value("b") == IntegerLiteral(0)
    # b reads as 0, so the conditional X on q[1] is not applied
    assert [ins.targets for ins in context.circuit.instructions] == [(0,)]
