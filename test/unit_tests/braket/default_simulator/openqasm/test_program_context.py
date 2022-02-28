from openqasm.ast import BoolType, FloatType, IntegerLiteral, IntType

from braket.default_simulator.openqasm.variable_transformer import SymbolTable, VariableTable


def test_symbol_table():
    int_8 = IntType(IntegerLiteral(8))
    int_16 = IntType(IntegerLiteral(16))
    float_8 = FloatType(IntegerLiteral(8))
    float_16 = FloatType(IntegerLiteral(16))
    boolean = BoolType()

    symbol_table = SymbolTable()
    symbol_table.add_symbol("x", int_8, True)
    symbol_table.add_symbol("y", float_16, False)
    symbol_table.add_symbol("z", boolean, False)

    assert symbol_table.get_type("x") == int_8
    assert symbol_table.get_type("y") == float_16
    assert symbol_table.get_type("z") == boolean

    assert symbol_table.get_const("x")
    assert not symbol_table.get_const("y")
    assert not symbol_table.get_const("z")

    symbol_table.push_scope()
    symbol_table.add_symbol("x", int_16, False)
    symbol_table.add_symbol("y", float_8, True)
    symbol_table.add_symbol("a", boolean, False)

    assert symbol_table.get_type("x") == int_16
    assert symbol_table.get_type("y") == float_8
    assert symbol_table.get_type("z") == boolean
    assert symbol_table.get_type("a") == boolean

    assert not symbol_table.get_const("x")
    assert symbol_table.get_const("y")
    assert not symbol_table.get_const("z")
    assert not symbol_table.get_const("a")

    symbol_table.pop_scope()
    assert symbol_table.get_type("x") == int_8
    assert symbol_table.get_type("y") == float_16
    assert symbol_table.get_type("z") == boolean

    assert symbol_table.get_const("x")
    assert not symbol_table.get_const("y")
    assert not symbol_table.get_const("z")


def test_variable_table():
    variable_table = VariableTable()
    variable_table.add_variable("x", 10)
    variable_table.add_variable("y", 1.34)
    variable_table.add_variable("z", False)

    assert variable_table.get_value("x") == 10
    assert variable_table.get_value("y") == 1.34
    assert not variable_table.get_value("z")

    variable_table.push_scope()
    variable_table.add_variable("x", 20)
    variable_table.add_variable("y", 2.68)
    variable_table.add_variable("a", "1001")

    assert variable_table.get_value("x") == 20
    assert variable_table.get_value("y") == 2.68
    assert not variable_table.get_value("z")
    assert variable_table.get_value("a") == "1001"

    variable_table.pop_scope()
    assert variable_table.get_value("x") == 10
    assert variable_table.get_value("y") == 1.34
    assert not variable_table.get_value("z")
