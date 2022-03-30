import pytest
from openqasm3.ast import BooleanLiteral, BoolType, FloatType, IntegerLiteral, IntType, RealLiteral

from braket.default_simulator.openqasm.program_context import ProgramContext, ScopedTable

boolean = BoolType()
int_8 = IntType(IntegerLiteral(8))
int_16 = IntType(IntegerLiteral(16))
float_8 = FloatType(IntegerLiteral(8))
float_16 = FloatType(IntegerLiteral(16))


def test_variable_declaration():
    context = ProgramContext()
    context.declare_variable("x", int_8, IntegerLiteral(10), True)
    context.declare_variable("y", float_16, RealLiteral(1.34), False)
    context.declare_variable("z", boolean, BooleanLiteral(False), False)

    def assert_scope_0():
        assert context.get_type("x") == int_8
        assert context.get_type("y") == float_16
        assert context.get_type("z") == boolean

        assert context.get_const("x")
        assert not context.get_const("y")
        assert not context.get_const("z")

        assert context.get_value("x") == IntegerLiteral(10)
        assert context.get_value("y") == RealLiteral(1.34)
        assert context.get_value("z") == BooleanLiteral(False)

    assert_scope_0()

    with context.enter_scope():
        context.declare_variable("x", int_16, IntegerLiteral(20), False)
        context.declare_variable("y", float_8, RealLiteral(2.68), True)
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
        assert context.get_value("y") == RealLiteral(2.68)
        assert context.get_value("z") == BooleanLiteral(False)
        assert context.get_value("a") == BooleanLiteral(True)

    assert_scope_0()


def test_repr():
    context = ProgramContext()
    context.declare_variable("x", int_8, IntegerLiteral(10), True)
    context.declare_variable("y", float_16, RealLiteral(1.34), False)
    context.declare_variable("z", boolean, BooleanLiteral(False), False)

    context.add_qubits("q")

    with context.enter_scope():
        context.declare_variable("x", int_16, IntegerLiteral(20), False)
        context.declare_variable("y", float_8, RealLiteral(2.68), True)
        context.declare_variable("a", boolean, BooleanLiteral(True), False)

        assert repr(context) == (
            """Symbols
SCOPE LEVEL 0
x	Symbol<IntType(span=None, size=IntegerLiteral(span=None, value=8)), const=True>
y	Symbol<FloatType(span=None, size=IntegerLiteral(span=None, value=16)), const=False>
z	Symbol<BoolType(span=None), const=False>
q	Symbol<<class 'openqasm3.ast.Identifier'>, const=False>
SCOPE LEVEL 1
x	Symbol<IntType(span=None, size=IntegerLiteral(span=None, value=16)), const=False>
y	Symbol<FloatType(span=None, size=IntegerLiteral(span=None, value=8)), const=True>
a	Symbol<BoolType(span=None), const=False>

Data
SCOPE LEVEL 0
x	IntegerLiteral(span=None, value=10)
y	RealLiteral(span=None, value=1.34)
z	BooleanLiteral(span=None, value=False)
q	Identifier(span=None, name='q')
SCOPE LEVEL 1
x	IntegerLiteral(span=None, value=20)
y	RealLiteral(span=None, value=2.68)
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

    with pytest.raises(KeyError, match="Undefined key: x"):
        del table["x"]
