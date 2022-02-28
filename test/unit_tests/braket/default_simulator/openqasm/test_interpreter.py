import numpy as np
import pytest
from openqasm.ast import IntegerLiteral, IntType, BitType, BooleanLiteral, UintType, FloatType, RealLiteral
from openqasm.parser.antlr.qasm_parser import parse

from braket.default_simulator.openqasm.program_context import ProgramContext
from braket.default_simulator.openqasm.variable_transformer import Interpreter


def test_bit_declaration():
    qasm = """
    bit single_uninitialized;
    bit single_initialized_int = 0;
    bit single_initialized_bool = true;
    bit[2] register_uninitialized;
    bit[2] register_initialized = "01";
    """
    program = parse(qasm)
    context: ProgramContext = Interpreter().visit(program)

    assert context.symbol_table.get_type("single_uninitialized") == BitType(None)
    assert context.symbol_table.get_type("single_initialized_int") == BitType(None)
    assert context.symbol_table.get_type("single_initialized_bool") == BitType(None)
    assert context.symbol_table.get_type("register_uninitialized") == BitType(IntegerLiteral(2))
    assert context.symbol_table.get_type("register_initialized") == BitType(IntegerLiteral(2))

    assert context.variable_table.get_value("single_uninitialized") is None
    assert context.variable_table.get_value("single_initialized_int") == BooleanLiteral(False)
    assert context.variable_table.get_value("single_initialized_bool") == BooleanLiteral(True)
    assert context.variable_table.get_value("register_uninitialized") is None
    assert context.variable_table.get_value("register_initialized") == IntegerLiteral(1)


def test_int_declaration():
    qasm = """
    int[8] uninitialized;
    int[8] pos = 10;
    int[5] neg = -4;
    int[3] pos_overflow = 5;
    int[3] neg_overflow = -6;
    """
    pos_overflow = "Integer overflow for value 5 and size 3."
    neg_overflow = "Integer overflow for value -6 and size 3."

    program = parse(qasm)
    with pytest.warns(UserWarning) as warn_info:
        context: ProgramContext = Interpreter().visit(program)

    assert context.symbol_table.get_type("uninitialized") == IntType(IntegerLiteral(8))
    assert context.symbol_table.get_type("pos") == IntType(IntegerLiteral(8))
    assert context.symbol_table.get_type("neg") == IntType(IntegerLiteral(5))
    assert context.symbol_table.get_type("pos_overflow") == IntType(IntegerLiteral(3))
    assert context.symbol_table.get_type("neg_overflow") == IntType(IntegerLiteral(3))

    assert context.variable_table.get_value("uninitialized") is None
    assert context.variable_table.get_value("pos") == IntegerLiteral(10)
    assert context.variable_table.get_value("neg") == IntegerLiteral(-4)
    assert context.variable_table.get_value("pos_overflow") == IntegerLiteral(1)
    assert context.variable_table.get_value("neg_overflow") == IntegerLiteral(-2)

    warnings = {(warn.category, warn.message.args[0]) for warn in warn_info}
    assert warnings == {
        (UserWarning, pos_overflow),
        (UserWarning, neg_overflow),
    }


def test_uint_declaration():
    qasm = """
    uint[8] uninitialized;
    uint[8] pos = 10;
    uint[3] pos_not_overflow = 5;
    uint[3] pos_overflow = 8;
    """
    pos_overflow = "Unsigned integer overflow for value 8 and size 3."

    program = parse(qasm)
    with pytest.warns(UserWarning, match=pos_overflow):
        context: ProgramContext = Interpreter().visit(program)

    assert context.symbol_table.get_type("uninitialized") == UintType(IntegerLiteral(8))
    assert context.symbol_table.get_type("pos") == UintType(IntegerLiteral(8))
    assert context.symbol_table.get_type("pos_not_overflow") == UintType(IntegerLiteral(3))
    assert context.symbol_table.get_type("pos_overflow") == UintType(IntegerLiteral(3))

    assert context.variable_table.get_value("uninitialized") is None
    assert context.variable_table.get_value("pos") == IntegerLiteral(10)
    assert context.variable_table.get_value("pos_not_overflow") == IntegerLiteral(5)
    assert context.variable_table.get_value("pos_overflow") == IntegerLiteral(0)


def test_float_declaration():
    qasm = """
    float[16] uninitialized;
    float[32] pos = 10;
    float[64] neg = -4.;
    float[128] precise = π;
    """
    program = parse(qasm)
    context: ProgramContext = Interpreter().visit(program)

    assert context.symbol_table.get_type("uninitialized") == FloatType(IntegerLiteral(16))
    assert context.symbol_table.get_type("pos") == FloatType(IntegerLiteral(32))
    assert context.symbol_table.get_type("neg") == FloatType(IntegerLiteral(64))
    assert context.symbol_table.get_type("precise") == FloatType(IntegerLiteral(128))

    assert context.variable_table.get_value("uninitialized") is None
    assert context.variable_table.get_value("pos") == RealLiteral(10)
    assert context.variable_table.get_value("neg") == RealLiteral(-4)
    assert context.variable_table.get_value("precise") == RealLiteral(np.pi)
