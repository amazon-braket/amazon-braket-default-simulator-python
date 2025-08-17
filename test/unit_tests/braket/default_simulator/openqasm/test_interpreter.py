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

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sympy
from sympy import Symbol

from braket.default_simulator import StateVectorSimulation
from braket.default_simulator.openqasm.interpreter import VerbatimBoxDelimiter
from braket.default_simulator.gate_operations import CX, GPhase, Hadamard, PauliX
from braket.default_simulator.gate_operations import PauliY as Y
from braket.default_simulator.gate_operations import RotX, U, Unitary
from braket.default_simulator.noise_operations import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    Kraus,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
    TwoQubitDephasing,
    TwoQubitDepolarizing,
)
from braket.default_simulator.observables import Hermitian, PauliY
from braket.default_simulator.openqasm._helpers.casting import (
    convert_bool_array_to_string,
    convert_string_to_bool_array,
)
from braket.default_simulator.openqasm.circuit import Circuit
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    AngleType,
    ArrayLiteral,
    ArrayType,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    FloatLiteral,
    FloatType,
    Identifier,
    IndexedIdentifier,
    IntegerLiteral,
    IntType,
    QuantumGate,
    QuantumGateDefinition,
    SymbolLiteral,
    UintType,
)
from braket.default_simulator.openqasm.program_context import QubitTable


def test_bit_declaration():
    qasm = """
    bit single_uninitialized;
    bit single_initialized_int = 0;
    bit single_initialized_bool = true;
    bit[2] register_uninitialized;
    bit[2] register_initialized = "01";
    """
    context = Interpreter().run(qasm)

    assert context.get_type("single_uninitialized") == BitType(None)
    assert context.get_type("single_initialized_int") == BitType(None)
    assert context.get_type("single_initialized_bool") == BitType(None)
    assert context.get_type("register_uninitialized") == BitType(IntegerLiteral(2))
    assert context.get_type("register_initialized") == BitType(IntegerLiteral(2))

    assert context.get_value("single_uninitialized") is None
    assert context.get_value("single_initialized_int") == BooleanLiteral(False)
    assert context.get_value("single_initialized_bool") == BooleanLiteral(True)
    assert context.get_value("register_uninitialized") == ArrayLiteral([None, None])
    assert context.get_value("register_initialized") == ArrayLiteral(
        [BooleanLiteral(False), BooleanLiteral(True)]
    )


def test_bool_declaration():
    qasm = """
    bool uninitialized;
    bool initialized_int = 0;
    bool initialized_bool = true;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == BoolType()
    assert context.get_type("initialized_int") == BoolType()
    assert context.get_type("initialized_bool") == BoolType()

    assert context.get_value("uninitialized") is None
    assert context.get_value("initialized_int") == BooleanLiteral(False)
    assert context.get_value("initialized_bool") == BooleanLiteral(True)


def test_int_declaration():
    qasm = """
    int[8] uninitialized;
    int[8] pos = 10;
    int[5] neg = -4;
    int[8] int_min = -128;
    int[8] int_max = 127;
    int[3] pos_overflow = 5;
    int[3] neg_overflow = -6;
    int no_size = 1e9;
    """
    pos_overflow = "Integer overflow for value 5 and size 3."
    neg_overflow = "Integer overflow for value -6 and size 3."

    with pytest.warns(UserWarning) as warn_info:
        context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == IntType(IntegerLiteral(8))
    assert context.get_type("pos") == IntType(IntegerLiteral(8))
    assert context.get_type("neg") == IntType(IntegerLiteral(5))
    assert context.get_type("pos_overflow") == IntType(IntegerLiteral(3))
    assert context.get_type("neg_overflow") == IntType(IntegerLiteral(3))
    assert context.get_type("no_size") == IntType(None)

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == IntegerLiteral(10)
    assert context.get_value("neg") == IntegerLiteral(-4)
    assert context.get_value("int_min") == IntegerLiteral(-128)
    assert context.get_value("int_max") == IntegerLiteral(127)
    assert context.get_value("pos_overflow") == IntegerLiteral(-3)
    assert context.get_value("neg_overflow") == IntegerLiteral(2)
    assert context.get_value("no_size") == IntegerLiteral(1_000_000_000)

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
    uint[3] neg_overflow = -1;
    uint no_size = 1e9;
    """
    pos_overflow = "Unsigned integer overflow for value 8 and size 3."

    with pytest.warns(UserWarning, match=pos_overflow):
        context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == UintType(IntegerLiteral(8))
    assert context.get_type("pos") == UintType(IntegerLiteral(8))
    assert context.get_type("pos_not_overflow") == UintType(IntegerLiteral(3))
    assert context.get_type("pos_overflow") == UintType(IntegerLiteral(3))
    assert context.get_type("neg_overflow") == UintType(IntegerLiteral(3))
    assert context.get_type("no_size") == UintType(None)

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == IntegerLiteral(10)
    assert context.get_value("pos_not_overflow") == IntegerLiteral(5)
    assert context.get_value("pos_overflow") == IntegerLiteral(0)
    assert context.get_value("neg_overflow") == IntegerLiteral(7)
    assert context.get_value("no_size") == IntegerLiteral(1_000_000_000)


def test_signed_int_cast():
    qasm = """
    uint[8] x0 = 255;
    int[8] x1 = x0;
    uint[8] x2 = x1;

    uint[8] y0 = 128;
    int[8] y1 = y0;
    uint[8] y2 = y1;

    int[3] z0 = "100";
    int[3] z1 = "111";
    """

    context = Interpreter().run(qasm)

    assert context.get_value("x0") == IntegerLiteral(255)
    assert context.get_value("x1") == IntegerLiteral(-1)
    assert context.get_value("x2") == IntegerLiteral(255)

    assert context.get_value("y0") == IntegerLiteral(128)
    assert context.get_value("y1") == IntegerLiteral(-128)
    assert context.get_value("y2") == IntegerLiteral(128)

    assert context.get_value("z0") == IntegerLiteral(-4)
    assert context.get_value("z1") == IntegerLiteral(-1)


def test_float_declaration():
    qasm = """
    float[16] uninitialized;
    float[32] pos = 10;
    float[64] neg = -4.2;
    float[64] precise = π;
    float unsized = π;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == FloatType(IntegerLiteral(16))
    assert context.get_type("pos") == FloatType(IntegerLiteral(32))
    assert context.get_type("neg") == FloatType(IntegerLiteral(64))
    assert context.get_type("precise") == FloatType(IntegerLiteral(64))
    assert context.get_type("unsized") == FloatType(None)

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == FloatLiteral(10)
    assert context.get_value("neg") == FloatLiteral(-4.2)
    assert context.get_value("precise") == FloatLiteral(np.pi)
    assert context.get_value("unsized") == FloatLiteral(np.pi)


def test_angle_declaration():
    qasm = """
    angle uninitialized;
    angle pos = 3.5 * π;
    angle neg = -4.5 * π;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == AngleType(None)
    assert context.get_type("pos") == AngleType(None)
    assert context.get_type("neg") == AngleType(None)

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == SymbolLiteral(1.5 * sympy.pi)
    assert context.get_value("neg") == SymbolLiteral(1.5 * sympy.pi)


def test_fixed_bit_angle_declaration():
    qasm = """
    angle[16] pos = 10;
    """
    with pytest.raises(ValueError):
        Interpreter().run(qasm)


def test_constant_declaration():
    qasm = """
    const float[16] const_tau = 2 * π;
    const int[8] const_one = 1;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("const_tau") == FloatType(IntegerLiteral(16))
    assert context.get_type("const_one") == IntType(IntegerLiteral(8))

    assert context.get_value("const_tau") == FloatLiteral(6.28125)
    assert context.get_value("const_one") == IntegerLiteral(1)

    assert context.get_const("const_tau")
    assert context.get_const("const_one")


def test_constant_immutability():
    qasm = """
    const int[8] const_one = 1;
    const_one = 2;
    """
    cannot_update = "Cannot update const value const_one"
    with pytest.raises(TypeError, match=cannot_update):
        Interpreter().run(qasm)


@pytest.mark.parametrize("index", ("", "[:]"))
def test_uninitialized_identifier(index):
    qasm = f"""
    bit[8] x;
    bit[8] y = x{index};
    """
    uninitialized_identifier = "Identifier 'x' is not initialized."
    with pytest.raises(NameError, match=uninitialized_identifier):
        Interpreter().run(qasm)


def test_assign_variable():
    qasm = """
    bit[8] og_bit = "10001000";
    bit[8] copy_bit = og_bit;

    int[10] og_int = 100;
    int[10] copy_int = og_int;

    uint[5] og_uint = 8;
    uint[5] copy_uint = og_uint;

    float[16] og_float = π;
    float[16] copy_float = og_float;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("copy_bit") == BitType(IntegerLiteral(8))
    assert context.get_type("copy_int") == IntType(IntegerLiteral(10))
    assert context.get_type("copy_uint") == UintType(IntegerLiteral(5))
    assert context.get_type("copy_float") == FloatType(IntegerLiteral(16))

    assert context.get_value("copy_bit") == convert_string_to_bool_array(
        BitstringLiteral(0b_10001000, 8)
    )
    assert context.get_value("copy_int") == IntegerLiteral(100)
    assert context.get_value("copy_uint") == IntegerLiteral(8)
    # notice the reduced precision compared to np.pi from float[16]
    assert context.get_value("copy_float") == FloatLiteral(3.140625)


def test_array_declaration():
    qasm = """
    array[uint[8], 2] row = {1, 2};
    array[int, 2] unsized_int = {1, 2};
    array[uint[8], 2, 2] multi_dim = {{1, 2}, {3, 4}};
    array[uint[8], 2, 2] by_ref = {row, row};
    array[uint[8], 1, 1, 1] with_expressions = {{{1 + 2}}};
    """
    context = Interpreter().run(qasm)

    assert context.get_type("row") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(2)]
    )
    assert context.get_type("unsized_int") == ArrayType(
        base_type=IntType(), dimensions=[IntegerLiteral(2)]
    )
    assert context.get_type("multi_dim") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(2), IntegerLiteral(2)]
    )
    assert context.get_type("by_ref") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(2), IntegerLiteral(2)]
    )
    assert context.get_type("with_expressions") == ArrayType(
        base_type=UintType(IntegerLiteral(8)),
        dimensions=[IntegerLiteral(1), IntegerLiteral(1), IntegerLiteral(1)],
    )

    assert context.get_value("row") == ArrayLiteral(
        [
            IntegerLiteral(1),
            IntegerLiteral(2),
        ]
    )
    assert context.get_value("unsized_int") == ArrayLiteral(
        [
            IntegerLiteral(1),
            IntegerLiteral(2),
        ]
    )
    assert context.get_value("multi_dim") == ArrayLiteral(
        [
            ArrayLiteral(
                [
                    IntegerLiteral(1),
                    IntegerLiteral(2),
                ]
            ),
            ArrayLiteral(
                [
                    IntegerLiteral(3),
                    IntegerLiteral(4),
                ]
            ),
        ]
    )
    assert context.get_value("by_ref") == ArrayLiteral(
        [
            ArrayLiteral(
                [
                    IntegerLiteral(1),
                    IntegerLiteral(2),
                ]
            ),
            ArrayLiteral(
                [
                    IntegerLiteral(1),
                    IntegerLiteral(2),
                ]
            ),
        ]
    )
    assert context.get_value("with_expressions") == ArrayLiteral(
        [ArrayLiteral([ArrayLiteral([IntegerLiteral(3)])])]
    )


@pytest.mark.parametrize(
    "qasm",
    (
        "array[uint[8], 2] row = {1, 2, 3};",
        "array[uint[8], 2, 2] multi_dim = {{1, 2}, {3, 4, 5}};",
        "array[uint[8], 2, 2] multi_dim = {{1, 2, 3}, {3, 4}};",
        "array[uint[8], 2, 2] multi_dim = {{1, 2}, {3, 4}, {5, 6}};",
    ),
)
def test_bad_array_size_declaration(qasm):
    bad_size = "Size mismatch between dimension of size 2 and values length 3"
    with pytest.raises(ValueError, match=bad_size):
        Interpreter().run(qasm)


def test_indexed_expression():
    qasm = """
    array[uint[8], 2, 2] multi_dim = {{1, 2}, {3, 4}};
    int[8] int_from_array = multi_dim[0, 1 * 1];
    array[int[8], 2] array_from_array = multi_dim[1];
    array[uint[8], 3] using_set = multi_dim[0][{1, 0, 1}];
    array[uint[8], 3, 2] using_set_multi_dim = multi_dim[{0, 1}][{1, 0, 1}];
    uint[4] fifteen = 15; // 1111
    uint[4] one = 1; // 0001
    bit[4] fifteen_b = fifteen[:];
    bit[4] one_b = one[0:3]; // 0001
    bit[3] trunc_b = one[0:-2]; // 000
    int[5] neg_fifteen = -15; // 10001
    int[5] neg_one = -1; // 11111
    bit[5] neg_fifteen_b = neg_fifteen[0:4];  // 10001
    bit[5] neg_one_b = neg_one[0:4];  // 11111
    bit[3] bit_slice = neg_fifteen_b[0:2:-1];  // 101
    array[uint[8], 2] one_three = multi_dim[:, 0];
    """
    context = Interpreter().run(qasm)

    assert context.get_type("multi_dim") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(2), IntegerLiteral(2)]
    )
    assert context.get_type("int_from_array") == IntType(IntegerLiteral(8))
    assert context.get_type("array_from_array") == ArrayType(
        base_type=IntType(IntegerLiteral(8)), dimensions=[IntegerLiteral(2)]
    )
    assert context.get_type("using_set") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(3)]
    )
    assert context.get_type("using_set_multi_dim") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(3), IntegerLiteral(2)]
    )

    assert context.get_value("multi_dim") == ArrayLiteral(
        [
            ArrayLiteral(
                [
                    IntegerLiteral(1),
                    IntegerLiteral(2),
                ]
            ),
            ArrayLiteral(
                [
                    IntegerLiteral(3),
                    IntegerLiteral(4),
                ]
            ),
        ]
    )
    assert context.get_value("int_from_array") == IntegerLiteral(2)
    assert context.get_value("array_from_array") == ArrayLiteral(
        [
            IntegerLiteral(3),
            IntegerLiteral(4),
        ]
    )
    assert context.get_value("using_set") == ArrayLiteral(
        [
            IntegerLiteral(2),
            IntegerLiteral(1),
            IntegerLiteral(2),
        ]
    )
    assert context.get_value("using_set_multi_dim") == ArrayLiteral(
        [
            ArrayLiteral(
                [
                    IntegerLiteral(3),
                    IntegerLiteral(4),
                ]
            ),
            ArrayLiteral(
                [
                    IntegerLiteral(1),
                    IntegerLiteral(2),
                ]
            ),
            ArrayLiteral(
                [
                    IntegerLiteral(3),
                    IntegerLiteral(4),
                ]
            ),
        ]
    )
    assert context.get_type("fifteen_b") == BitType(IntegerLiteral(4))
    assert context.get_type("one_b") == BitType(IntegerLiteral(4))
    assert context.get_type("trunc_b") == BitType(IntegerLiteral(3))
    assert context.get_type("neg_fifteen_b") == BitType(IntegerLiteral(5))
    assert context.get_type("neg_one_b") == BitType(IntegerLiteral(5))
    assert context.get_type("bit_slice") == BitType(IntegerLiteral(3))
    assert context.get_value("fifteen_b") == convert_string_to_bool_array(
        BitstringLiteral(0b_1111, 4)
    )
    assert context.get_value("one_b") == convert_string_to_bool_array(BitstringLiteral(0b_0001, 4))
    assert context.get_value("trunc_b") == convert_string_to_bool_array(BitstringLiteral(0b_000, 3))
    assert context.get_value("neg_fifteen_b") == convert_string_to_bool_array(
        BitstringLiteral(0b_10001, 5)
    )
    assert context.get_value("neg_one_b") == convert_string_to_bool_array(
        BitstringLiteral(0b_11111, 5)
    )
    assert context.get_value("bit_slice") == convert_string_to_bool_array(
        BitstringLiteral(0b_101, 3)
    )
    assert context.get_value("one_three") == ArrayLiteral(
        values=[IntegerLiteral(1), IntegerLiteral(3)]
    )


def test_reset_qubit():
    qasm = """
    qubit q;
    reset q;
    """
    no_reset = "Reset not supported"
    with pytest.raises(NotImplementedError, match=no_reset):
        Interpreter().run(qasm)


def test_for_loop():
    qasm = """
    int[8] x = 0;
    int[8] y = -100;
    int[8] ten = 10;

    for uint[8] i in [0:2:ten - 3] {
        x += i;
    }

    for int[8] i in {2, 4, 6} {
        y += i;
    }
    """
    context = Interpreter().run(qasm)

    assert context.get_value("x") == IntegerLiteral(sum((0, 2, 4, 6)))
    assert context.get_value("y") == IntegerLiteral(sum((-100, 2, 4, 6)))


def test_while_loop():
    qasm = """
    int[8] x = 0;
    int[8] i = 0;

    while (i < 7) {
        x += i;
        i += 1;
    }
    """
    context = Interpreter().run(qasm)
    assert context.get_value("x") == IntegerLiteral(sum(range(7)))


def test_indexed_identifier():
    qasm = """
    uint[8] one;
    uint[8] another_one;
    array[uint[8], 2] twos;
    array[uint[8], 2] threes;
    bit[4] fz;

    array[uint[8], 2, 2] empty;

    fz = "0000";
    array[uint[8], 2, 2] multi_dim = {{0, 0}, {0, 0}};
    array[uint[8], 4] single_dim = {0, 0, 0, 0};
    array[bit[4], 2, 2] multi_dim_bit = {{fz, fz}, {fz, fz}};

    multi_dim[0, 0] = 1;
    one = multi_dim[0, 0];

    array[uint[8], 1] one_one = {1};
    multi_dim[0, 1:1] = one_one;
    another_one = multi_dim[0, 0];

    array[uint[8], 2] two_twos = {2, 2};
    multi_dim[1, :] = two_twos;
    twos = multi_dim[1, :];

    array[uint[8], 2] two_threes = {3, 3};
    multi_dim[:, 0] = two_threes;
    threes = multi_dim[:, 0];

    fz[1:2:3] = "11";

    array[bit[2], 2] two_elevens = {"11", "11"};
    multi_dim_bit[:, 0, 1:2:3] = two_elevens;
    multi_dim_bit[0][1][0] = true;
    
    array[uint[4], 2, 2] range_4 = {{1, 2}, {3, 4}};
    bit[4] two = range_4[0, 1, :];
    """
    context = Interpreter().run(qasm)
    assert context.get_value("one") == IntegerLiteral(1)
    assert context.get_value("another_one") == IntegerLiteral(1)
    assert context.get_value("twos") == ArrayLiteral(
        [
            IntegerLiteral(2),
            IntegerLiteral(2),
        ]
    )
    assert context.get_value("threes") == ArrayLiteral(
        [
            IntegerLiteral(3),
            IntegerLiteral(3),
        ]
    )
    assert context.get_value("fz") == ArrayLiteral(
        [
            BooleanLiteral(False),
            BooleanLiteral(True),
            BooleanLiteral(False),
            BooleanLiteral(True),
        ]
    )
    assert context.get_value("multi_dim_bit") == ArrayLiteral(
        [
            ArrayLiteral(
                [
                    ArrayLiteral(
                        [
                            BooleanLiteral(False),
                            BooleanLiteral(True),
                            BooleanLiteral(False),
                            BooleanLiteral(True),
                        ]
                    ),
                    ArrayLiteral(
                        [
                            BooleanLiteral(True),
                            BooleanLiteral(False),
                            BooleanLiteral(False),
                            BooleanLiteral(False),
                        ]
                    ),
                ]
            ),
            ArrayLiteral(
                [
                    ArrayLiteral(
                        [
                            BooleanLiteral(False),
                            BooleanLiteral(True),
                            BooleanLiteral(False),
                            BooleanLiteral(True),
                        ]
                    ),
                    ArrayLiteral(
                        [
                            BooleanLiteral(False),
                            BooleanLiteral(False),
                            BooleanLiteral(False),
                            BooleanLiteral(False),
                        ]
                    ),
                ]
            ),
        ]
    )
    assert context.get_value("empty") == ArrayLiteral(
        [
            ArrayLiteral([None, None]),
            ArrayLiteral([None, None]),
        ]
    )
    assert context.get_value("two") == ArrayLiteral(
        [BooleanLiteral(False), BooleanLiteral(False), BooleanLiteral(True), BooleanLiteral(False)]
    )


def test_get_bits_unsized_int():
    qasm = """
    int x = 2;
    bit from_x = x[0];
    """
    no_bit_ops = "Cannot perform bit operations on an unsized integer"
    with pytest.raises(TypeError, match=no_bit_ops):
        Interpreter().run(qasm)


def test_update_bits_int():
    qasm = """
    int[4] x = 2;
    x[3] = 1;
    int[4] y = 2;
    y[0] = 1;
    uint[4] z = 2;
    z[0] = 1;
    """
    context = Interpreter().run(qasm)
    assert context.get_value("x") == IntegerLiteral(3)
    assert context.get_value("y") == IntegerLiteral(-6)
    assert context.get_value("z") == IntegerLiteral(10)


def test_update_bits_int_unsized():
    qasm = """
    int x = 2;
    x[3] = 1;
    """
    no_bit_ops = "Cannot perform bit operations on an unsized integer"
    with pytest.raises(TypeError, match=no_bit_ops):
        Interpreter().run(qasm)


def test_gate_def():
    qasm = """
    float[64] my_pi = π;
    gate x0 a { U(π, 0, my_pi) a; }
    gate x1(mp) c { U(π, 0, mp) c; }
    gate x2(p) a, b {
        x0 b;
        x1(p) a;
        x1(my_pi) a;
        U(1, 2, p) b;
    }
    """
    context = Interpreter().run(qasm)

    assert context.get_gate_definition("x0") == QuantumGateDefinition(
        name=Identifier("x0"),
        arguments=[],
        qubits=[Identifier("a")],
        body=[
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    SymbolLiteral(sympy.pi),
                    IntegerLiteral(0),
                    FloatLiteral(np.pi),
                ],
                qubits=[Identifier("a")],
            )
        ],
    )
    assert context.get_gate_definition("x1") == QuantumGateDefinition(
        name=Identifier("x1"),
        arguments=[Identifier("mp")],
        qubits=[Identifier("c")],
        body=[
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    SymbolLiteral(sympy.pi),
                    IntegerLiteral(0),
                    Identifier("mp"),
                ],
                qubits=[Identifier("c")],
            )
        ],
    )
    assert context.get_gate_definition("x2") == QuantumGateDefinition(
        name=Identifier("x2"),
        arguments=[Identifier("p")],
        qubits=[Identifier("a"), Identifier("b")],
        body=[
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    SymbolLiteral(sympy.pi),
                    IntegerLiteral(0),
                    FloatLiteral(np.pi),
                ],
                qubits=[Identifier("b")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    SymbolLiteral(sympy.pi),
                    IntegerLiteral(0),
                    Identifier("p"),
                ],
                qubits=[Identifier("a")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    SymbolLiteral(sympy.pi),
                    IntegerLiteral(0),
                    FloatLiteral(np.pi),
                ],
                qubits=[Identifier("a")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    IntegerLiteral(1),
                    IntegerLiteral(2),
                    Identifier("p"),
                ],
                qubits=[Identifier("b")],
            ),
        ],
    )


def test_gate_undef():
    qasm = """
    gate x a { undef_y a; }
    gate undef_y a { U(π, π/2, π/2) a; }
    """
    undefined_gate = "Gate undef_y is not defined."
    with pytest.raises(ValueError, match=undefined_gate):
        Interpreter().run(qasm)


def test_gate_call():
    qasm = """
    float[64] my_pi = π;
    gate x2(p) a { U(π, 0, p) a; }

    qubit q1;
    qubit q2;
    qubit[2] qs;

    U(π, 0, my_pi) q1;
    x q2;
    x2(my_pi) qs[1];
    
    // overwrite x gate
    gate x a { y a; }
    x qs;

    """
    circuit = Interpreter().build_circuit(qasm)
    expected_circuit = Circuit(
        instructions=[
            U((0,), np.pi, 0, np.pi, ()),
            PauliX((1,)),
            U((3,), np.pi, 0, np.pi, ()),
            Y((2,)),
            Y((3,)),
        ]
    )
    assert circuit == expected_circuit


def test_gate_inv():
    qasm = """
    gate rand_u_1 a { U(1, 2, 3) a; }
    gate rand_u_2 a { U(2, 3, 4) a; }
    gate rand_u_3 a { inv @ U(3, 4, 5) a; }

    gate both a {
        rand_u_1 a;
        rand_u_2 a;
    }
    gate both_inv a {
        inv @ both a;
    }
    gate all_3 a {
        rand_u_1 a;
        rand_u_2 a;
        rand_u_3 a;
    }
    gate all_3_inv a {
        inv @ inv @ inv @ all_3 a;
    }

    gate apply_phase a {
        gphase(1);
    }

    gate apply_phase_inv a {
        inv @ gphase(1);
    }

    qubit q;

    both q;
    both_inv q;

    all_3 q;
    all_3_inv q;

    apply_phase q;
    apply_phase_inv q;

    U(1, 2, 3) q;
    inv @ U(1, 2, 3) q;

    s q;
    inv @ s q;

    t q;
    inv @ t q;
    """
    circuit = Interpreter().build_circuit(qasm)
    coeff = np.linalg.multi_dot(
        [
            instruction.matrix
            for instruction in circuit.instructions
            if isinstance(instruction, GPhase)
        ]
    )[0][0]
    collapsed = np.linalg.multi_dot(
        [
            instruction.matrix
            for instruction in circuit.instructions
            if not isinstance(instruction, GPhase)
        ]
    )
    assert np.allclose(coeff * collapsed, np.eye(2**circuit.num_qubits))


def test_gate_ctrl():
    qasm = """
    int[8] two = 2;
    gate x a { U(π, 0, π) a; }
    gate cx c, a {
        ctrl @ x c, a;
    }
    gate ccx_1 c1, c2, a {
        ctrl @ ctrl @ x c1, c2, a;
    }
    gate ccx_2 c1, c2, a {
        ctrl(two) @ x c1, c2, a;
    }
    gate ccx_3 c1, c2, a {
        ctrl @ cx c1, c2, a;
    }

    qubit q1;
    qubit q2;
    qubit q3;
    qubit q4;
    qubit q5;

    // doesn't flip q2
    cx q1, q2;
    // flip q1
    x q1;
    // flip q2
    cx q1, q2;
    // doesn't flip q3, q4, q5
    ccx_1 q1, q4, q3;
    ccx_2 q1, q3, q4;
    ccx_3 q1, q3, q5;
    // flip q3, q4, q5;
    ccx_1 q1, q2, q3;
    ccx_2 q1, q2, q4;
    ccx_2 q1, q2, q5;
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(5, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(simulation.state_vector, np.array([0] * 31 + [1]))


def test_gate_ctrl_global():
    qasm = """
    qubit q1;
    qubit q2;

    h q1;
    h q2;
    ctrl @ s q1, q2;
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(2, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(
        simulation.state_vector,
        [0.5, 0.5, 0.5, 0.5j],
    )


def test_neg_gate_ctrl():
    qasm = """
    int[8] two = 2;
    gate x a { U(π, 0, π) a; }
    gate cx c, a {
        negctrl @ x c, a;
    }
    gate ccx_1 c1, c2, a {
        negctrl @ negctrl @ x c1, c2, a;
    }
    gate ccx_2 c1, c2, a {
        negctrl(two) @ x c1, c2, a;
    }
    gate ccx_3 c1, c2, a {
        negctrl @ cx c1, c2, a;
    }

    qubit q1;
    qubit q2;
    qubit q3;
    qubit q4;
    qubit q5;

    x q1;
    x q2;
    x q3;
    x q4;
    x q5;

    // doesn't flip q2
    cx q1, q2;
    // flip q1
    x q1;
    // flip q2
    cx q1, q2;
    // doesn't flip q3, q4, q5
    ccx_1 q1, q4, q3;
    ccx_2 q1, q3, q4;
    ccx_3 q1, q3, q5;
    // flip q3, q4, q5;
    ccx_1 q1, q2, q3;
    ccx_2 q1, q2, q4;
    ccx_3 q1, q2, q5;
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(5, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(simulation.state_vector, np.array([1] + [0] * 31))


def test_pow():
    qasm = """
    int[8] two = 2;
    gate x a { U(π, 0, π) a; }
    gate cx c, a {
        pow(1) @ ctrl @ x c, a;
    }
    gate cxx_1 c, a {
        pow(two) @ cx c, a;
    }
    gate cxx_2 c, a {
        pow(1./2.) @ pow(4) @ cx c, a;
    }
    gate cxxx c, a {
        pow(1) @ pow(two) @ cx c, a;
    }

    qubit q1;
    qubit q2;
    qubit q3;
    qubit q4;
    qubit q5;

    pow(1./2) @ x q1;   // half flip
    pow(1/2.) @ x q1;  // half flip
    cx q1, q2;          // flip
    cxx_1 q1, q3;       // don't flip
    cxx_2 q1, q4;       // don't flip
    cnot q1, q5;        // flip
    x q3;               // flip
    x q4;               // flip
    pow(1/2) @ x q5;    // don't flip
    
    s q1;               // sqrt z
    s q1;               // again
    inv @ z q1;         // inv z
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(5, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(simulation.state_vector, np.array([0] * 31 + [1]))


def test_measurement_noop_does_not_raise_exceptions():
    qasm = """
    qubit[2] q;
    bit[1] c;
    measure q[0];
    c = measure q[1];
    """
    Interpreter().run(qasm)


def test_gphase():
    qasm = """
    qubit[2] qs;

    int[8] two = 2;

    gate x a { U(π, 0, π) a; }
    gate cx c, a { ctrl @ x c, a; }
    gate phase c, a {
        gphase(π/2);
        pow(1) @ ctrl(two) @ gphase(π) c, a;
    }
    gate h a { U(π/2, 0, π) a; }

    h qs[0];
    cx qs[0], qs[1];
    phase qs[0], qs[1];

    gphase(π);
    inv @ gphase(π / 2);
    negctrl @ ctrl @ gphase(2 * π) qs[0], qs[1];
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(2, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(simulation.state_vector, [-1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def test_no_neg_ctrl_phase():
    qasm = """
    gate bad_phase a {
        negctrl @ gphase(π/2);
    }
    """
    no_negctrl = "negctrl modifier undefined for gphase operation"
    with pytest.raises(ValueError, match=no_negctrl):
        Interpreter().run(qasm)


def test_if():
    qasm = """
    int[8] two = 2;
    bit[3] m = "000";

    if (two + 1) {
        m[0] = 1;
    } else {
        m[1] = 1;
    }

    if (!bool(two - 2)) {
        m[2] = 1;
    }
    """
    context = Interpreter().run(qasm)
    assert convert_bool_array_to_string(context.get_value("m")) == "101"


def test_include_stdgates():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        Path("stdgates.inc").touch()
        with open("stdgates.inc", "w", encoding="utf-8") as f:
            f.write(
                """
            // OpenQASM 3 standard gate library

// phase gate
gate p(λ) a { ctrl @ gphase(λ) a; }

// Pauli gate: bit-flip or NOT gate
gate x a { U(π, 0, π) a; }
// Pauli gate: bit and phase flip
gate y a { U(π, π/2, π/2) a; }
 // Pauli gate: phase flip
gate z a { p(π) a; }

// Clifford gate: Hadamard
gate h a { U(π/2, 0, π) a; }
// Clifford gate: sqrt(Z) or S gate
gate s a { pow(1/2) @ z a; }
// Clifford gate: inverse of sqrt(Z)
gate sdg a { inv @ pow(1/2) @ z a; }

// sqrt(S) or T gate
gate t a { pow(1/2) @ s a; }
// inverse of sqrt(S)
gate tdg a { inv @ pow(1/2) @ s a; }

// sqrt(NOT) gate
gate sx a { pow(1/2) @ x a; }

// Rotation around X-axis
gate rx(θ) a { U(θ, -π/2, π/2) a; }
// rotation around Y-axis
gate ry(θ) a { U(θ, 0, 0) a; }
// rotation around Z axis
gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }

// controlled-NOT
gate cx a, b { ctrl @ x a, b; }
// controlled-Y
gate cy a, b { ctrl @ y a, b; }
// controlled-Z
gate cz a, b { ctrl @ z a, b; }
// controlled-phase
gate cp(λ) a, b { ctrl @ p(λ) a, b; }
// controlled-rx
gate crx(θ) a, b { ctrl @ rx(θ) a, b; }
// controlled-ry
gate cry(θ) a, b { ctrl @ ry(θ) a, b; }
// controlled-rz
gate crz(θ) a, b { ctrl @ rz(θ) a, b; }
// controlled-H
gate ch a, b { ctrl @ h a, b; }

// swap
gate swap a, b { cx a, b; cx b, a; cx a, b; }

// Toffoli
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
// controlled-swap
gate cswap a, b, c { ctrl @ swap a, b, c; }

// four parameter controlled-U gate with relative phase γ
gate cu(θ, φ, λ, γ) a, b { p(γ) a; ctrl @ U(θ, φ, λ) a, b; }

// Gates for OpenQASM 2 backwards compatibility
// CNOT
gate CX a, b { ctrl @ U(π, 0, π) a, b; }
// phase gate
gate phase(λ) q { U(0, 0, λ) q; }
// controlled-phase
gate cphase(λ) a, b { ctrl @ phase(λ) a, b; }
// identity or idle gate
gate id a { U(0, 0, 0) a; }
// IBM Quantum experience gates
gate u1(λ) q { U(0, 0, λ) q; }
gate u2(φ, λ) q { gphase(-(φ+λ)/2); U(π/2, φ, λ) q; }
gate u3(θ, φ, λ) q { gphase(-(φ+λ)/2); U(θ, φ, λ) q; }
            """
            )
        qasm = """
        OPENQASM 3;
        include "stdgates.inc";
        """
        context = Interpreter().run(qasm)

        assert {
            "p",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "sx",
            "rx",
            "ry",
            "rz",
            "cx",
            "cy",
            "cz",
            "cp",
            "crx",
            "cry",
            "crz",
            "ch",
            "swap",
            "ccx",
            "cswap",
            "cu",
            "CX",
            "phase",
            "cphase",
            "id",
            "u1",
            "u2",
            "u3",
        }.issubset(context.gate_table.current_scope.keys())
        os.chdir("..")


def test_adder():
    qasm = """

input uint[4] a_in;
input uint[4] b_in;

gate majority a, b, c {
    cnot c, b;
    cnot c, a;
    ccnot a, b, c;
}

gate unmaj a, b, c {
    ccnot a, b, c;
    cnot c, a;
    cnot a, b;
}

qubit cin;
qubit[4] a;
qubit[4] b;
qubit cout;

// set input states
for int[8] i in [0: 3] {
  if(bool(a_in[i])) x a[i];
  if(bool(b_in[i])) x b[i];
}

// add a to b, storing result in b
majority cin, b[3], a[3];
for int[8] i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
cnot a[0], cout;
for int[8] i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
unmaj cin, b[3], a[3];

// measure results
//ans[0] = measure cout;
//ans[1:4] = measure b[0:3];
"""
    circuit = Interpreter().build_circuit(qasm, {"a_in": 1, "b_in": 15})
    simulation = StateVectorSimulation(10, 1, 1)
    simulation.evolve(circuit.instructions)
    assert simulation.probabilities[0b_0_0001_0000_1] == 1


def test_assignment_operators():
    qasm = """
    int[16] x;
    bit[4] xs;

    x = 0;
    xs = "0000";

    x += 1; // 1
    x *= 2; // 2
    x /= 2; // 1
    x -= 5; // -4

    xs[2:] |= "11";
    """
    context = Interpreter().run(qasm)
    assert context.get_value("x") == IntegerLiteral(-4)
    assert convert_bool_array_to_string(context.get_value("xs")) == "0011"


def test_resolve_result_type():
    qasm = """
    float[16] small_pi;
    float[16] one_point_five;
    int[8] int_not_bool;

    int[8] one = 1;
    float[16] half = 0.5;
    bit t = true;

    small_pi = π + one - 1;
    one_point_five = one + half;
    int_not_bool = t + 1;
    """
    context = Interpreter().run(qasm)
    assert context.get_value("small_pi") == SymbolLiteral(sympy.pi)
    assert context.get_value("one_point_five") == FloatLiteral(1.5)
    assert context.get_value("int_not_bool") == IntegerLiteral(2)


def test_bit_operators():
    qasm = """
    bit[4] and;
    bit[4] or;
    bit[4] xor;
    bit[4] lshift;
    bit[4] rshift;
    bit[4] flip;
    bit gt;
    bit lt;
    bit ge;
    bit le;
    bit eq;
    bit neq;
    bit not;
    bit not_zero;

    bit[4] x = "0101";
    bit[4] y = "1100";

    and = x & y;
    or = x | y;
    xor = x ^ y;
    lshift = x << 2;
    rshift = y >> 2;
    flip = ~x;
    gt = x > y;
    lt = x < y;
    ge = x >= y;
    le = x <= y;
    eq = x == y;
    neq = x != y;
    not = !x;
    not_zero = !(x << 4);
    """
    context = Interpreter().run(qasm)
    assert convert_bool_array_to_string(context.get_value("and")) == "0100"
    assert convert_bool_array_to_string(context.get_value("or")) == "1101"
    assert convert_bool_array_to_string(context.get_value("xor")) == "1001"
    assert convert_bool_array_to_string(context.get_value("lshift")) == "0100"
    assert convert_bool_array_to_string(context.get_value("rshift")) == "0011"
    assert convert_bool_array_to_string(context.get_value("flip")) == "1010"
    assert context.get_value("gt") == BooleanLiteral(False)
    assert context.get_value("lt") == BooleanLiteral(True)
    assert context.get_value("ge") == BooleanLiteral(False)
    assert context.get_value("le") == BooleanLiteral(True)
    assert context.get_value("eq") == BooleanLiteral(False)
    assert context.get_value("neq") == BooleanLiteral(True)
    assert context.get_value("not") == BooleanLiteral(False)
    assert context.get_value("not_zero") == BooleanLiteral(True)


@pytest.mark.parametrize("in_int", (0, 1, -2, 5))
def test_input(in_int):
    qasm = """
    input int[8] in_int;
    input bit[8] in_bit;
    int[8] doubled;

    doubled = in_int * 2;
    """
    in_bit = "10110010"
    context = Interpreter().run(
        qasm,
        inputs={
            "in_int": in_int,
            "in_bit": in_bit,
        },
    )
    assert context.get_value("doubled") == IntegerLiteral(in_int * 2)
    assert context.get_value("in_bit") == convert_string_to_bool_array(
        BitstringLiteral(value=0b10110010, width=8)
    )


def test_output():
    qasm = """
    output int[8] out_int;
    """
    output_not_supported = "Output not supported"
    with pytest.raises(NotImplementedError, match=output_not_supported):
        Interpreter().run(qasm)


def test_missing_input():
    qasm = """
    input int[8] in_int;
    int[8] doubled;

    doubled = in_int * 2;
    qubit q;
    rx(doubled) q;
    """
    circuit = Interpreter().build_circuit(qasm)
    for instruction in circuit.instructions:
        print(
            f"{type(instruction).__name__}({getattr(instruction, '_angle', None)}) "
            f"{', '.join(map(str, instruction._targets))}"
        )
    assert circuit.instructions == [RotX([0], 2.0 * Symbol("in_int"))]


@pytest.mark.parametrize("bad_index", ("[0:1][0][0]", "[0][0][1]", "[0, 1][0]", "[0:1, 1][0]"))
def test_qubit_multidim(bad_index):
    qasm = f"""
    qubit q;
    U(1, 0, 1) q{bad_index};
    """
    multi_dim_qubit = "Cannot index multiple dimensions for qubits."
    with pytest.raises(IndexError, match=multi_dim_qubit):
        Interpreter().run(qasm)


def test_qubit_multi_dim_index():
    """this verifies that interpreter handles q[0, 1] correctly"""
    multi_dim_qubit = "Cannot index multiple dimensions for qubits."
    with pytest.raises(IndexError, match=multi_dim_qubit):
        QubitTable().get_by_identifier(
            IndexedIdentifier(
                Identifier("q"),
                [[IntegerLiteral(0), IntegerLiteral(1)]],
            )
        )


def test_physical_qubits():
    qasm = """
    h $0;
    cnot $0, $1;
    """
    circuit = Interpreter().build_circuit(qasm)
    expected_circuit = Circuit(
        instructions=[
            Hadamard((0,)),
            CX((0, 1)),
        ]
    )
    assert circuit == expected_circuit


@pytest.mark.parametrize("bad_op", ("x - y", "-x"))
def test_invalid_op(bad_op):
    qasm = f"""
    bit[4] x = "0001";
    bit[4] y = "1010";

    bit[4] z = {bad_op};
    """
    invalid_op = "Invalid operator - for ArrayLiteral"
    with pytest.raises(TypeError, match=invalid_op):
        Interpreter().run(qasm)


def test_bad_bit_declaration():
    qasm = """
    bit[4] x = "00010";
    """
    invalid_op = re.escape(
        "Invalid array to cast to bit register of size 4: "
        "ArrayLiteral(span=None, values=[BooleanLiteral(span=None, value=False), "
        "BooleanLiteral(span=None, value=False), BooleanLiteral(span=None, value=False), "
        "BooleanLiteral(span=None, value=True), BooleanLiteral(span=None, value=False)])."
    )
    with pytest.raises(ValueError, match=invalid_op):
        Interpreter().run(qasm)


def test_bad_float_declaration():
    qasm = """
    float[4] x = π;
    """
    invalid_float = "Float size must be one of {16, 32, 64}."
    with pytest.raises(ValueError, match=invalid_float):
        Interpreter().run(qasm)


def test_bad_update_values_declaration_non_array():
    qasm = """
    bit[4] x = "0000";
    x[0:1] = 1;
    """
    invalid_value = "Must assign Array type to slice"
    with pytest.raises(ValueError, match=invalid_value):
        Interpreter().run(qasm)


def test_bad_update_values_declaration_size_mismatch():
    qasm = """
    bit[4] x = "0000";
    x[0:1] = "111";
    """
    invalid_value = "Dimensions do not match: 2, 3"
    with pytest.raises(ValueError, match=invalid_value):
        Interpreter().run(qasm)


def test_gate_qubit_reg():
    qasm = """
    qubit[3] qs;
    qubit q;

    x qs[{0, 2}];
    h q;
    cphaseshift(1) qs, q;
    phaseshift(-2) q;
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(4, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(
        simulation.state_vector,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            0,
            0,
        ],
    )


def test_gate_qubit_reg_size_mismatch():
    qasm = """
    qubit[3] qs;
    qubit q;

    x qs[{0, 2}];
    ctrl @ rx(π/2) qs, qs[0:1];
    """
    size_mismatch = "Qubit registers must all be the same length."
    with pytest.raises(ValueError, match=size_mismatch):
        Interpreter().run(qasm)


# noqa: E501
def test_unitary_pragma():
    qasm = """
    qubit[3] q;

    x q[0];
    h q[1];

    // unitary pragma for t gate
    #pragma braket unitary([[1.0, 0], [0, 0.70710678 + 0.70710678im]]) q[0]
    ti q[0];

    // unitary pragma for h gate (with phase shift)
    #pragma braket unitary([[0.70710678 im, 0.70710678im], [0 - -0.70710678im, -0.0 - 0.70710678im]]) q[1]
    gphase(-π/2) q[1];
    h q[1];

    // unitary pragma for ccnot gate
    #pragma braket unitary([[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0, 1.0, 0]]) q
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(3, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(
        simulation.state_vector,
        [0, 0, 0, 0, 0.70710678, 0, 0, 0.70710678],
    )


def test_bad_unitary_pragma():
    qasm = """
    qubit q;
    #pragma braket unitary([[1.0, 0, 1], [0, 0.70710678 + 0.70710678im]]) q
    """
    invalid_matrix = "Not a valid square matrix"
    with pytest.raises(TypeError, match=invalid_matrix):
        Interpreter().run(qasm)


def test_verbatim_pragma():
    with_verbatim = """
    OPENQASM 3.0;
    bit[2] b;
    qubit[2] q;
    #pragma braket verbatim
    box{
    cnot q[0], q[1];
    cnot q[0], q[1];
    rx(1.57) q[0];
    }
    b[0] = measure q[0];
    b[1] = measure q[1];
    """
    circuit = Interpreter().build_circuit(with_verbatim)
    sim_w_verbatim = StateVectorSimulation(2, 1, 1)
    sim_w_verbatim.evolve(circuit.instructions)

    without_verbatim = """
    OPENQASM 3.0;
    bit[2] b;
    qubit[2] q;
    box{
    cnot q[0], q[1];
    cnot q[0], q[1];
    rx(1.57) q[0];
    }
    b[0] = measure q[0];
    b[1] = measure q[1];
    """
    circuit = Interpreter().build_circuit(without_verbatim)
    sim_wo_verbatim = StateVectorSimulation(2, 1, 1)
    sim_wo_verbatim.evolve(circuit.instructions)

    assert np.allclose(
        sim_w_verbatim.state_vector,
        sim_wo_verbatim.state_vector,
    )


def test_unsupported_pragma():
    qasm = """
    qubit q;
    #pragma braket abcd
    box{
    rx(1.57) q;
    }
    """
    unsupported_pragma = "Pragma 'braket abcd' is not supported"
    with pytest.raises(NotImplementedError, match=unsupported_pragma):
        Interpreter().run(qasm)


def test_subroutine():
    qasm = """
    const int[8] n = 4;
    def parity(bit[n] cin) -> bit {
      bit c = false;
      for int[8] i in [0: n - 1] {
        c ^= cin[i];
      }
      return c;
    }

    bit[4] c = "1011";
    bit p = parity(c);
    """
    context = Interpreter().run(qasm)
    assert context.get_value("p") == BooleanLiteral(True)


def test_undefined_subroutine():
    qasm = """
    const int[8] n = 4;
    bit[4] c = "1011";
    bit p = parity(c);
    """
    subroutine_undefined = "Subroutine parity is not defined."
    with pytest.raises(NameError, match=subroutine_undefined):
        Interpreter().run(qasm)


def test_void_subroutine():
    qasm = """
    def flip(qubit q) {
      x q;
    }

    qubit[2] qs;
    flip(qs[0]);
    """
    circuit = Interpreter().build_circuit(qasm)
    simulation = StateVectorSimulation(2, 1, 1)
    simulation.evolve(circuit.instructions)
    assert np.allclose(
        simulation.state_vector,
        [0, 0, 1, 0],
    )


def test_array_ref_subroutine():
    qasm = """
    int[16] total_1;
    int[16] total_2;

    def sum(const array[int[8], #dim = 1] arr) -> int[16] {
        int[16] size = sizeof(arr);
        int[16] x = 0;
        for int[8] i in [0:size - 1] {
            x += arr[i];
        }
        return x;
    }

    array[int[8], 5] array_1 = {1, 2, 3, 4, 5};
    array[int[8], 10] array_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    total_1 = sum(array_1);
    total_2 = sum(array_2);
    """
    context = Interpreter().run(qasm)
    assert context.get_value("total_1") == IntegerLiteral(15)
    assert context.get_value("total_2") == IntegerLiteral(55)


def test_subroutine_array_reference_mutation():
    qasm = """
    def mutate_array(mutable array[int[8], #dim = 1] arr) {
        int[16] size = sizeof(arr);
        for int[8] i in [0:size - 1] {
            arr[i] = 0;
        }
    }

    array[int[8], 5] array_1 = {1, 2, 3, 4, 5};
    array[int[8], 10] array_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    array[int[8], 10] array_3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    mutate_array(array_1);
    mutate_array(array_2);
    mutate_array(array_3[4:2:-1]);
    """
    context = Interpreter().run(qasm)
    assert context.get_value("array_1") == ArrayLiteral([IntegerLiteral(0)] * 5)
    assert context.get_value("array_2") == ArrayLiteral([IntegerLiteral(0)] * 10)
    assert context.get_value("array_3") == ArrayLiteral(
        [IntegerLiteral(x) for x in (1, 2, 3, 4, 0, 6, 0, 8, 0, 10)]
    )


def test_subroutine_array_reference_const_mutation():
    qasm = """
    def mutate_array(const array[int[8], #dim = 1] arr) {
        int[16] size = sizeof(arr);
        for int[8] i in [0:size - 1] {
            arr[i] = 0;
        }
    }

    array[int[8], 5] array_1 = {1, 2, 3, 4, 5};
    mutate_array(array_1);
    """
    cannot_mutate = "Cannot update const value arr"
    with pytest.raises(TypeError, match=cannot_mutate):
        Interpreter().run(qasm)


def test_subroutine_classical_passed_by_value():
    qasm = """
    def classical(bit[4] bits) {
        bits[0] = 1;
        return bits;
    }

    bit[4] before = "0000";
    bit[4] after = classical(before);
    """
    context = Interpreter().run(qasm)
    assert context.get_value("before") == ArrayLiteral(
        [BooleanLiteral(False), BooleanLiteral(False), BooleanLiteral(False), BooleanLiteral(False)]
    )
    assert context.get_value("after") == ArrayLiteral(
        [BooleanLiteral(True), BooleanLiteral(False), BooleanLiteral(False), BooleanLiteral(False)]
    )


def test_builtin_functions():
    qasm = """
        const float[64] arccos_result = arccos(1);
        const float[64] arcsin_result = arcsin(1);
        const float[64] arctan_result = arctan(1);
        const int[64] ceiling_result = ceiling(π);
        const float[64] cos_result = cos(1);
        const float[64] exp_result = exp(2);
        const int[64] floor_result = floor(π);
        const float[64] log_result = log(ℇ);
        const int[64] mod_int_result = mod(4, 3);
        const float[64] mod_float_result = mod(5.2, 2.5);
        const int[64] popcount_bit_result = popcount("1001110");
        const int[64] popcount_int_result = popcount(78);
        // parser gets confused by pow
        // const int[64] pow_int_result = pow(3, 3);
        // const float[64] pow_float_result = pow(2.5, 2.5);
        // add rotl, rotr
        const float[64] sin_result = sin(1);
        const float[64] sqrt_result = sqrt(2);
        const float[64] tan_result = tan(1);
        """
    context = Interpreter().run(qasm)
    assert context.get_value("arccos_result") == FloatLiteral(np.arccos(1))
    assert context.get_value("arcsin_result") == FloatLiteral(np.arcsin(1))
    assert context.get_value("arctan_result") == FloatLiteral(np.arctan(1))
    assert context.get_value("ceiling_result") == IntegerLiteral(4)
    assert context.get_value("cos_result") == FloatLiteral(np.cos(1))
    assert context.get_value("exp_result") == FloatLiteral(np.exp(2))
    assert context.get_value("floor_result") == IntegerLiteral(3)
    assert context.get_value("log_result") == FloatLiteral(1)
    assert context.get_value("mod_int_result") == IntegerLiteral(1)
    assert context.get_value("mod_float_result") == FloatLiteral(5.2 % 2.5)
    assert context.get_value("popcount_bit_result") == IntegerLiteral(4)
    assert context.get_value("popcount_int_result") == IntegerLiteral(4)
    assert context.get_value("sin_result") == FloatLiteral(np.sin(1))
    assert context.get_value("sqrt_result") == FloatLiteral(np.sqrt(2))
    assert context.get_value("tan_result") == FloatLiteral(np.tan(1))


def test_builtin_functions_symbolic():
    qasm = """
        input float x;
        input float y;
        
        rx(x) $0;
        rx(arccos(x)) $0;
        rx(arcsin(x)) $0;
        rx(arctan(x)) $0;
        rx(ceiling(x)) $0;
        rx(cos(x)) $0;
        rx(exp(x)) $0;
        rx(floor(x)) $0;
        rx(log(x)) $0;
        rx(mod(x, y)) $0;
        rx(sin(x)) $0;
        rx(sqrt(x)) $0;
        rx(tan(x)) $0;
        """
    circuit = Interpreter().build_circuit(qasm)
    assert circuit == Circuit(
        [
            RotX((0,), Symbol("x")),
            RotX((0,), sympy.acos(Symbol("x"))),
            RotX((0,), sympy.asin(Symbol("x"))),
            RotX((0,), sympy.atan(Symbol("x"))),
            RotX((0,), sympy.ceiling(Symbol("x"))),
            RotX((0,), sympy.cos(Symbol("x"))),
            RotX((0,), sympy.exp(Symbol("x"))),
            RotX((0,), sympy.floor(Symbol("x"))),
            RotX((0,), sympy.log(Symbol("x"))),
            RotX((0,), Symbol("x") % Symbol("y")),
            RotX((0,), sympy.sin(Symbol("x"))),
            # sympy evalf changes sqrt to power in handle_parameter_value
            RotX((0,), Symbol("x") ** 0.5),
            RotX((0,), sympy.tan(Symbol("x"))),
        ]
    )
    with_inputs = Interpreter().build_circuit(qasm, inputs={"x": 1, "y": 2})
    simulation = StateVectorSimulation(1, 1, 1)
    simulation.evolve(with_inputs.instructions)


def test_noise():
    qasm = """
    qubit[2] qs;

    #pragma braket noise bit_flip(.5) qs[1]
    #pragma braket noise phase_flip(.5) qs[0]
    #pragma braket noise pauli_channel(.1, .2, .3) qs[0]
    #pragma braket noise depolarizing(.5) qs[0]
    #pragma braket noise two_qubit_depolarizing(.9) qs
    #pragma braket noise two_qubit_depolarizing(.7) qs[1], qs[0]
    #pragma braket noise two_qubit_dephasing(.6) qs
    #pragma braket noise amplitude_damping(.2) qs[0]
    #pragma braket noise generalized_amplitude_damping(.2, .3)  qs[1]
    #pragma braket noise phase_damping(.4) qs[0]
    #pragma braket noise kraus([[0.9486833im, 0], [0, 0.9486833im]], [[0, 0.31622777], [0.31622777, 0]]) qs[0]
    #pragma braket noise kraus([[0.9486832980505138, 0, 0, 0], [0, 0.9486832980505138, 0, 0], [0, 0, 0.9486832980505138, 0], [0, 0, 0, 0.9486832980505138]], [[0, 0.31622776601683794, 0, 0], [0.31622776601683794, 0, 0, 0], [0, 0, 0, 0.31622776601683794], [0, 0, 0.31622776601683794, 0]]) qs[{1, 0}]
    """
    circuit = Interpreter().build_circuit(qasm)
    assert circuit.instructions == [
        BitFlip([1], 0.5),
        PhaseFlip([0], 0.5),
        PauliChannel([0], 0.1, 0.2, 0.3),
        Depolarizing([0], 0.5),
        TwoQubitDepolarizing((0, 1), 0.9),
        TwoQubitDepolarizing([1, 0], 0.7),
        TwoQubitDephasing([0, 1], 0.6),
        AmplitudeDamping([0], 0.2),
        GeneralizedAmplitudeDamping([1], 0.2, 0.3),
        PhaseDamping([0], 0.4),
        Kraus(
            [0],
            [
                np.array([[0.9486833j, 0], [0, 0.9486833j]]),
                np.array([[0, 0.31622777], [0.31622777, 0]]),
            ],
        ),
        Kraus(
            [1, 0],
            [
                np.eye(4) * np.sqrt(0.9),
                np.kron([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]) * np.sqrt(0.1),
            ],
        ),
    ]


@pytest.mark.parametrize(
    "qasm",
    (
        "const int x = 4;",
        "qubit[2] q; h q[0:1];",
        "gate my_x q { x q; }",
        "qubit[2] q; ctrl @ x q[0], q[1];",
        "qubit q; if (1) { x q; }",
        "qubit q; for int i in [0:10] { x q; }",
        "while (false) {}",
        "def subroutine() {}",
        "def subroutine() {} subroutine();",
    ),
)
def test_advanced_language_features(qasm, caplog):
    Interpreter().run(qasm, inputs={"x": 1})
    assert re.match(
        (
            "WARNING.*"
            "This program uses OpenQASM language features that may "
            "not be supported on QPUs or on-demand simulators\\.\n"
        ),
        caplog.text,
    )


def test_basis_rotation():
    qasm = """
    qubit[3] q;
    i q;
    
    #pragma braket result expectation z(q[2]) @ x(q[0])
    #pragma braket result variance x(q[0]) @ y(q[1])
    #pragma braket result sample x(q[0])
    """
    circuit = Interpreter().build_circuit(qasm)
    assert circuit.basis_rotation_instructions == [
        Hadamard([0]),
        Unitary([1], PauliY._diagonalizing_matrix),
    ]


def test_basis_rotation_identity():
    qasm = """
    qubit[3] q;
    i q;
    
    #pragma braket result expectation z(q[2]) @ x(q[0])
    #pragma braket result variance x(q[0]) @ y(q[1])
    #pragma braket result sample i(q[0])
    """
    circuit = Interpreter().build_circuit(qasm)
    assert circuit.basis_rotation_instructions == [
        Hadamard([0]),
        Unitary([1], PauliY._diagonalizing_matrix),
    ]


def test_basis_rotation_hermitian():
    qasm = """
    qubit[3] q;
    i q;
    #pragma braket result expectation x(q[2])
    // # noqa: E501
    #pragma braket result expectation hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0:1]
    // # noqa: E501
    #pragma braket result expectation x(q[2]) @ hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0:1]
    """
    circuit = Interpreter().build_circuit(qasm)
    array = np.array(
        [
            [-6, 2 + 1j, -3, -5 + 2j],
            [2 - 1j, 0, 2 - 1j, -5 + 4j],
            [-3, 2 + 1j, 0, -4 + 3j],
            [-5 - 2j, -5 - 4j, -4 - 3j, -6],
        ]
    )
    hermitian = Hermitian(array, targets=[0, 1])
    assert circuit.basis_rotation_instructions == [
        Hadamard([2]),
        *hermitian.diagonalizing_gates(),
    ]


@pytest.mark.parametrize(
    "qasm, expected",
    [
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[2] q;",
                    "h q[0];",
                    "h q[1];",
                    "b[0] = measure q[0];",
                ]
            ),
            ([0], [0]),
        ),
        (
            "\n".join(
                [
                    "bit[3] b;",
                    "qubit[3] q;",
                    "b = measure q;",
                ]
            ),
            ([0, 1, 2], [0, 1, 2]),
        ),
        (
            "\n".join(
                [
                    "bit[2] b;",
                    "qubit[2] q;",
                    "h q[0];",
                    "h q[1];",
                    "b[0:1] = measure q[0:1];",
                ]
            ),
            ([0, 1], [0, 1]),
        ),
        (
            "\n".join(
                [
                    "bit[3] b;",
                    "qubit[3] q;",
                    "h q[0];",
                    "cnot q[0], q[1];",
                    "cnot q[1], q[2];",
                    "b[0] = measure q[0];",
                    "b[2] = measure q[1];",
                    "b[1] = measure q[2];",
                ]
            ),
            ([0, 1, 2], [0, 2, 1]),
        ),
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[3] q;",
                    "h q[0];",
                    "h q[1];",
                    "cnot q[1], q[2];",
                    "b[{2, 1}] = measure q[{0, 2}];",
                ]
            ),
            ([0, 2], [2, 1]),
        ),
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "h $0;",
                    "cnot $0, $1;",
                    "b[0] = measure $0;",
                ]
            ),
            ([0], [0]),
        ),
        (
            "\n".join(
                [
                    "qubit[5] q;",
                    "for int i in [0:2] {",
                    "   measure q[i];",
                    "}",
                ]
            ),
            ([0, 1, 2], [0, 1, 2]),
        ),
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[3] q;",
                    "h q[0];",
                    "h q[1];",
                    "cnot q[1], q[2];",
                    "measure q[1];",
                    "measure q[0];",
                ]
            ),
            ([1, 0], [0, 1]),
        ),
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[2] q;",
                    "b[0] = measure q[1:5];",
                ]
            ),
            ([1], [0]),
        ),
    ],
)
def test_measurement(qasm, expected):
    circuit = Interpreter().build_circuit(qasm)
    assert circuit.measured_qubits == expected[0]
    assert circuit.target_classical_indices == expected[1]


@pytest.mark.parametrize(
    "qasm, expected",
    [
        (
            "\n".join(
                [
                    "bit[3] b;",
                    "qubit[2] q;",
                    "h q[0];",
                    "cnot q[0], q[1];",
                    "b[2] = measure q[1];",
                    "b[0] = measure q[0];",
                    "b[1] = measure q[0];",
                ]
            ),
            "Qubit 0 is already measured or captured.",
        ),
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[1] q;",
                    "h q[0];",
                    "b[0] = measure q[0];",
                    "measure q;",
                ]
            ),
            "Qubit 0 is already measured or captured.",
        ),
    ],
)
def test_measurement_exceptions(qasm, expected):
    with pytest.raises(ValueError, match=expected):
        Interpreter().build_circuit(qasm)


def test_measure_invalid_qubit():
    qasm = """
    bit[1] b;
    qubit[1] q;
    h q[0];
    measure x;
    """
    expected = "Undefined key: x"
    with pytest.raises(KeyError, match=expected):
        Interpreter().build_circuit(qasm)


@pytest.mark.parametrize(
    "qasm, expected",
    [
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[1] q;",
                    "b[0] = measure q[5];",
                ]
            ),
            "qubit register index `5` out of range for qubit register of length 1 `q`.",
        ),
        (
            "\n".join(
                [
                    "bit[1] b;",
                    "qubit[2] q;",
                    "b[0] = measure q[{1, 5}];",
                ]
            ),
            "qubit register index `5` out of range for qubit register of length 2 `q`.",
        ),
    ],
)
def test_measure_qubit_out_of_range(qasm, expected):
    with pytest.raises(IndexError, match=expected):
        Interpreter().build_circuit(qasm)


@pytest.mark.parametrize(
    "qasm,error_message",
    [
        (
            "\n".join(["OPENQASM 3.0;bit[2] b;", "qubit[1] q;", "b[{0, 1}] = measure q[0];"]),
            re.escape(
                "Number of qubits (1) does not match number of provided classical targets (2)"
            ),
        ),
        (
            "\n".join(["OPENQASM 3.0;bit[2] b;", "qubit[2] q;", "b[0][2] = measure q[1];"]),
            re.escape("Multi-Dimensional indexing not supported for classical registers."),
        ),
    ],
)
def test_invalid_measurement_with_classical_indices(qasm, error_message):
    with pytest.raises(ValueError, match=error_message):
        Interpreter().build_circuit(qasm)

def test_verbatim_box_start():
    vbs = VerbatimBoxDelimiter.START_VERBATIM
    assert isinstance(vbs, VerbatimBoxDelimiter)
    assert vbs.value == "StartVerbatim"
    assert vbs.name == "START_VERBATIM"
    assert vbs.qubit_count == 0

def test_verbatim_box_end():
    vbs = VerbatimBoxDelimiter.END_VERBATIM
    assert isinstance(vbs, VerbatimBoxDelimiter)
    assert vbs.value == "EndVerbatim"
    assert vbs.name =="END_VERBATIM"
    assert vbs.qubit_count == 0

def test_verbatim_box():
    qasm_with_verbatim = """
        OPENQASM 3.0;
        #pragma braket verbatim
        box {
        h $0;
        cnot $0, $1;
        }
    """
    context= Interpreter().run(qasm_with_verbatim)
   
    is_verbatim  = context.in_verbatim_box
    assert isinstance(context.circuit.instructions[0], Hadamard)
    assert isinstance(context.circuit.instructions[1], CX)
    assert isinstance(is_verbatim, bool)
    assert is_verbatim == False

def test_verbatim_wo_box():
    qasm_without_box = """
        OPENQASM 3.0;
        #pragma braket verbatim
        h $0;
    """
    with pytest.raises(ValueError, match="braket verbatim pragma must be followed by a box statement"):
        Interpreter().run(qasm_without_box)
