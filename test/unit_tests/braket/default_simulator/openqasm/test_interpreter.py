import re
from typing import Dict
from unittest.mock import Mock, call

import numpy as np
import pytest
from openqasm3.ast import (
    ArrayLiteral,
    ArrayType,
    BitType,
    BooleanLiteral,
    FloatType,
    Identifier,
    IndexedIdentifier,
    IntegerLiteral,
    IntType,
    QuantumGate,
    QuantumGateDefinition,
    RealLiteral,
    StringLiteral,
    UintType,
)

from braket.default_simulator.openqasm import data_manipulation
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.program_context import ProgramContext, QubitTable


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

    with pytest.warns(UserWarning) as warn_info:
        context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == IntType(IntegerLiteral(8))
    assert context.get_type("pos") == IntType(IntegerLiteral(8))
    assert context.get_type("neg") == IntType(IntegerLiteral(5))
    assert context.get_type("pos_overflow") == IntType(IntegerLiteral(3))
    assert context.get_type("neg_overflow") == IntType(IntegerLiteral(3))

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == IntegerLiteral(10)
    assert context.get_value("neg") == IntegerLiteral(-4)
    assert context.get_value("pos_overflow") == IntegerLiteral(1)
    assert context.get_value("neg_overflow") == IntegerLiteral(-2)

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

    with pytest.warns(UserWarning, match=pos_overflow):
        context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == UintType(IntegerLiteral(8))
    assert context.get_type("pos") == UintType(IntegerLiteral(8))
    assert context.get_type("pos_not_overflow") == UintType(IntegerLiteral(3))
    assert context.get_type("pos_overflow") == UintType(IntegerLiteral(3))

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == IntegerLiteral(10)
    assert context.get_value("pos_not_overflow") == IntegerLiteral(5)
    assert context.get_value("pos_overflow") == IntegerLiteral(0)


def test_float_declaration():
    qasm = """
    float[16] uninitialized;
    float[32] pos = 10;
    float[64] neg = -4.;
    float[128] precise = π;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("uninitialized") == FloatType(IntegerLiteral(16))
    assert context.get_type("pos") == FloatType(IntegerLiteral(32))
    assert context.get_type("neg") == FloatType(IntegerLiteral(64))
    assert context.get_type("precise") == FloatType(IntegerLiteral(128))

    assert context.get_value("uninitialized") is None
    assert context.get_value("pos") == RealLiteral(10)
    assert context.get_value("neg") == RealLiteral(-4)
    assert context.get_value("precise") == RealLiteral(np.pi)


def test_constant_declaration():
    qasm = """
    const float[16] const_tau = 2 * π;
    const int[8] const_one = 1;
    """
    context = Interpreter().run(qasm)

    assert context.get_type("const_tau") == FloatType(IntegerLiteral(16))
    assert context.get_type("const_one") == IntType(IntegerLiteral(8))

    assert context.get_value("const_tau") == RealLiteral(6.28125)
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

    assert context.get_value("copy_bit") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("10001000")
    )
    assert context.get_value("copy_int") == IntegerLiteral(100)
    assert context.get_value("copy_uint") == IntegerLiteral(8)
    # notice the reduced precision compared to np.pi from float[16]
    assert context.get_value("copy_float") == RealLiteral(3.140625)


def test_array_declaration():
    qasm = """
    array[uint[8], 2] row = {1, 2};
    array[uint[8], 2, 2] multi_dim = {{1, 2}, {3, 4}};
    array[uint[8], 2, 2] by_ref = {row, row};
    array[uint[8], 1, 1, 1] with_expressions = {{{1 + 2}}};
    """
    context = Interpreter().run(qasm)

    assert context.get_type("row") == ArrayType(
        base_type=UintType(IntegerLiteral(8)), dimensions=[IntegerLiteral(2)]
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
    assert context.get_value("fifteen_b") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("1111")
    )
    assert context.get_value("one_b") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("0001")
    )
    assert context.get_value("trunc_b") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("000")
    )
    assert context.get_value("neg_fifteen_b") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("10001")
    )
    assert context.get_value("neg_one_b") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("11111")
    )
    assert context.get_value("bit_slice") == data_manipulation.convert_string_to_bool_array(
        StringLiteral("101")
    )
    assert context.get_value("one_three") == ArrayLiteral(
        values=[IntegerLiteral(1), IntegerLiteral(3)]
    )


def test_reset_qubit():
    qasm = """
    qubit q;
    qubit[2] qs;
    qubit[2] qs2;
    qubit[5] qs5;

    int[8] two = 2;

    reset q;
    reset qs;
    reset qs2[0];
    reset qs5[:two:4];
    reset qs5[{4, 1, 0}];
    """
    mocked_context = ProgramContext()
    reset_qubits_mock = Mock()
    mocked_context.quantum_simulation.reset_qubits = reset_qubits_mock
    Interpreter(mocked_context).run(qasm)

    reset_qubits_mock.assert_has_calls(
        (
            call((0,)),
            call((1, 2)),
            call((3,)),
            call((5, 7, 9)),
            call((9, 6, 5)),
        )
    )


def test_for_loop():
    qasm = """
    gate x a { U(π, 0, π) a; }
    qubit[8] q;

    int[8] ten = 10;
    bit[8] m1;
    bit[8] m2;

    reset q;

    for i in [0:2:ten - 3] {
        x q[i];
    }
    m1 = measure q;

    for i in {2, 4, 6} {
        reset q[i];
    }
    m2 = measure q;
    """
    context = Interpreter().run(qasm, shots=1)

    assert np.array_equal(
        context.shot_data["m1"],
        ("10101010",),
    )
    assert np.array_equal(
        context.shot_data["m2"],
        ("10000000",),
    )


def test_for_loop_shots():
    qasm = """
    output bit[10] b;
    int[8] ten = 10;

    for i in [0:ten - 1] {
        b[i] = 1;
    }

    for i in [4: -1: 0] {
        b[i] = 0;
    }
    """
    context = Interpreter().run(qasm, shots=10)
    assert shot_data_is_equal(
        context.shot_data,
        {
            "b": np.full(10, "0000011111"),
        },
    )


def test_while_loop():
    qasm = """
    output bit[10] b;
    b = "0000000000";
    int[8] ten = 10;
    int[8] i = 0;

    while (i < 7) {
        b[i] = 1;
        i = i + 1;
    }
    """
    context = Interpreter().run(qasm, shots=10)
    assert shot_data_is_equal(
        context.shot_data,
        {
            "b": np.full(10, "1111111000"),
        },
    )


def test_indexed_identifier():
    qasm = """
    output uint[8] one;
    output uint[8] another_one;
    output array[uint[8], 2] twos;
    output array[uint[8], 2] threes;
    output bit[4] fz;

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
    """
    context = Interpreter().run(qasm, shots=0)
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


def test_gate_def():
    qasm = """
    float[128] my_pi = π;
    gate x a { U(π, 0, my_pi) a; }
    gate x1(mp) c { U(π, 0, mp) c; }
    gate x2(p) a, b {
        x b;
        x1(p) a;
        x1(my_pi) a;
        U(1, 2, p) b;
    }
    """
    context = Interpreter().run(qasm)

    assert context.get_gate_definition("x") == QuantumGateDefinition(
        name=Identifier("x"),
        arguments=[],
        qubits=[Identifier("a")],
        body=[
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    RealLiteral(np.pi),
                    IntegerLiteral(0),
                    RealLiteral(np.pi),
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
                    RealLiteral(np.pi),
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
                    RealLiteral(np.pi),
                    IntegerLiteral(0),
                    RealLiteral(np.pi),
                ],
                qubits=[Identifier("b")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    RealLiteral(np.pi),
                    IntegerLiteral(0),
                    Identifier("p"),
                ],
                qubits=[Identifier("a")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    RealLiteral(np.pi),
                    IntegerLiteral(0),
                    RealLiteral(np.pi),
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
    gate x a { y a; }
    gate y a { U(π, π/2, π/2) a; }
    """
    undefined_gate = "Gate y is not defined."
    with pytest.raises(ValueError, match=undefined_gate):
        Interpreter().run(qasm)


def test_gate_call():
    qasm = """
    float[128] my_pi = π;
    gate x a { U(π, 0, my_pi) a; }
    gate x2(p) a { U(π, 0, p) a; }

    qubit q1;
    qubit q2;
    qubit[2] qs;

    reset q1;
    reset q2;
    reset qs;

    U(π, 0, my_pi) q1;
    x q2;
    x2(my_pi) qs[0];
    """
    context = Interpreter().run(qasm)

    assert np.allclose(
        context.quantum_simulation.state_vector, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    )


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

    reset q;

    both q;
    both_inv q;

    all_3 q;
    all_3_inv q;

    apply_phase q;
    apply_phase_inv q;
    """
    context = Interpreter().run(qasm)

    assert np.allclose(context.quantum_simulation.state_vector, [1, 0])


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

    reset q1;
    reset q2;
    reset q3;
    reset q4;
    reset q5;

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
    context = Interpreter().run(qasm)
    assert np.allclose(
        context.quantum_simulation.state_vector,
        [0] * 31 + [1],
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

    reset q1;
    reset q2;
    reset q3;
    reset q4;
    reset q5;

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
    ccx_2 q1, q2, q5;
    """
    context = Interpreter().run(qasm)
    assert np.allclose(
        context.quantum_simulation.state_vector,
        [1] + [0] * 31,
    )


def test_measurement():
    qasm = """
    qubit q;
    qubit[2] qs;
    qubit[2] qs2;
    qubit[5] qs5;

    int[8] two = 2;

    gate x a { U(π, 0, π) a; }

    reset q;
    reset qs;
    reset qs2;
    reset qs5;

    x q;
    x qs[0];
    x qs5[0];
    x qs5[2];

    bit mq;
    bit[2] mqs;
    bit[1] mqs2_0;
    bit[two] mqs5;

    mq = measure q;
    mqs = measure qs;
    mqs2_0 = measure qs2[0];
    mqs5 = measure qs5[:two:3];
    """
    context = Interpreter().run(qasm, shots=1)

    measurements = {
        "mq": "1",
        "mqs": "10",
        "mqs2_0": "0",
        "mqs5": "11",
    }

    for bit, result in measurements.items():
        assert np.array_equal(
            context.shot_data[bit],
            [result],
        )


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

    reset qs;

    h qs[0];
    cx qs[0], qs[1];
    phase qs[0], qs[1];

    gphase(π);
    inv @ gphase(π / 2);
    negctrl @ ctrl @ gphase(2 * π) qs[0], qs[1];
    """
    context = Interpreter().run(qasm)
    print(context.quantum_simulation.state_vector)
    assert np.allclose(
        context.quantum_simulation.state_vector, [-1j / np.sqrt(2), 0, 0, 1j / np.sqrt(2)]
    )


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
    bit[3] m;

    qubit[3] qs;
    reset qs;

    gate x a { U(π, 0, π) a; }

    if (two + 1) {
        x qs[0];
    } else {
        x qs[1];
    }

    if (!bool(two - 2)) {
        x qs[2];
    }

    m = measure qs;
    """
    context = Interpreter().run(qasm, shots=1)
    assert np.array_equal(
        context.shot_data["m"],
        ["101"],
    )


def test_include_stdgates(stdgates):
    qasm = """
    OPENQASM 3;
    include "stdgates.inc";
    """
    context = Interpreter().run(qasm)

    assert np.array_equal(
        list(context.gate_table.current_scope.keys()),
        [
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
        ],
    )


def shot_data_is_equal(s1: Dict, s2: Dict):
    assert set(s1.keys()) == set(s2.keys())
    for key in s1.keys():
        if not np.array_equal(s1[key], s2[key]):
            return False
    return True


def test_adder(adder):
    context = Interpreter().run_file("adder.qasm", 10, {"a_in": 1, "b_in": 15})

    assert shot_data_is_equal(
        context.shot_data,
        {"ans": np.full(10, "10000")},
    )


def test_shots(stdgates):
    bell_qasm = """
    include "stdgates.inc";

    qubit[3] q;
    bit[3] c;
    bit d = true;

    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];

    c = measure q;
    """
    context = Interpreter().run(bell_qasm, shots=10)

    assert len(context.shot_data["c"]) == 10
    for result in context.shot_data["c"]:
        assert result in ("000", "111")


@pytest.mark.parametrize(
    "used_qubit",
    ("qs", "qs[0]", "qs[:]", "qs[0:]", "qs[:-1]", "qs[-2]", "qs[0:1:1]", "qs[-2:1:-1]"),
)
def test_cannot_reset_used_analytic(stdgates, used_qubit):
    qasm = f"""
    include "stdgates.inc";

    qubit[2] qs;

    reset qs;

    h qs[0];
    reset qs[1];
    reset {used_qubit};
    """
    Interpreter().run(qasm, shots=2)
    cannot_reset_qubit = re.escape(
        f"Cannot reset qubit(s) '{used_qubit}' since doing so would collapse "
        "the wave function in a shots=0 simulation."
    )
    with pytest.raises(ValueError, match=cannot_reset_qubit):
        Interpreter().run(qasm)


def test_cannot_measure_analytic():
    qasm = """
    qubit q;
    reset q;
    measure q;
    """
    Interpreter().run(qasm, shots=2)
    cannot_measure_analytic = re.escape(
        "Measurement operation not supported for analytic shots=0 simulation."
    )
    with pytest.raises(ValueError, match=cannot_measure_analytic):
        Interpreter().run(qasm)


def test_assignment_operators():
    qasm = """
    output int[16] x;
    output bit[4] xs;

    x = 0;
    xs = "0000";

    x += 1; // 1
    x *= 2; // 2
    x /= 2; // 1
    x -= 5; // -4

    xs[2:] |= "11";
    """
    context = Interpreter().run(qasm, shots=1)
    assert shot_data_is_equal(
        context.shot_data,
        {
            "x": [-4],
            "xs": ["0011"],
        },
    )


def test_resolve_result_type():
    qasm = """
    output float[16] small_pi;
    output float[16] one_point_five;
    output int[8] int_not_bool;

    int[8] one = 1;
    float[16] half = 0.5;
    bit t = true;

    small_pi = π + one - 1;
    one_point_five = one + half;
    int_not_bool = t + 1;
    """
    context = Interpreter().run(qasm, shots=1)
    assert shot_data_is_equal(
        context.shot_data,
        {
            "small_pi": [3.140625],
            "one_point_five": [1.5],
            "int_not_bool": [2],
        },
    )


def test_bit_operators():
    qasm = """
    output bit[4] and;
    output bit[4] or;
    output bit[4] xor;
    output bit[4] lshift;
    output bit[4] rshift;
    output bit[4] flip;
    output bit gt;
    output bit lt;
    output bit ge;
    output bit le;
    output bit eq;
    output bit neq;
    output bit not;
    output bit not_zero;

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
    context = Interpreter().run(qasm, shots=1)
    assert shot_data_is_equal(
        context.shot_data,
        {
            "and": ["0100"],
            "or": ["1101"],
            "xor": ["1001"],
            "lshift": ["0100"],
            "rshift": ["0011"],
            "flip": ["1010"],
            "gt": [False],
            "lt": [True],
            "ge": [False],
            "le": [True],
            "eq": [False],
            "neq": [True],
            "not": [False],
            "not_zero": [True],
        },
    )


@pytest.mark.parametrize("in_int", (0, 1, -2, 5))
def test_input(in_int):
    qasm = """
    input int[8] in_int;
    output int[8] doubled;

    doubled = in_int * 2;
    """
    context = Interpreter().run(qasm, shots=1, inputs={"in_int": in_int})
    assert shot_data_is_equal(context.shot_data, {"doubled": [in_int * 2]})


def test_missing_input():
    qasm = """
    input int[8] in_int;
    output int[8] doubled;

    doubled = in_int * 2;
    """
    missing_input = "Missing input variable 'in_int'."
    with pytest.raises(NameError, match=missing_input):
        Interpreter().run(qasm, shots=1)


@pytest.mark.parametrize("bad_index", ("[0:1][0][0]", "[0][0][1]", "[0, 1][0]", "[0:1, 1][0]"))
def test_qubit_multidim(bad_index):
    qasm = f"""
    qubit q;
    reset q{bad_index};
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


@pytest.mark.parametrize("bad_op", ("x - y", "-x"))
def test_invalid_op(bad_op):
    qasm = f"""
    bit[4] x = "0001";
    bit[4] y = "1010";

    bit[4] z = {bad_op};
    """
    invalid_op = "Invalid operator - for ArrayLiteral"
    with pytest.raises(TypeError, match=invalid_op):
        Interpreter().run(qasm, shots=1)


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
        Interpreter().run(qasm, shots=1)


def test_bad_float_declaration():
    qasm = """
    float[4] x = π;
    """
    invalid_float = "Float size must be one of {16, 32, 64, 128}."
    with pytest.raises(ValueError, match=invalid_float):
        Interpreter().run(qasm, shots=1)


def test_bad_update_values_declaration_non_array():
    qasm = """
    bit[4] x = "0000";
    x[0:1] = 1;
    """
    invalid_value = "Must assign Array type to slice"
    with pytest.raises(ValueError, match=invalid_value):
        Interpreter().run(qasm, shots=1)


def test_bad_update_values_declaration_size_mismatch():
    qasm = """
    bit[4] x = "0000";
    x[0:1] = "111";
    """
    invalid_value = "Dimensions do not match: 2, 3"
    with pytest.raises(ValueError, match=invalid_value):
        Interpreter().run(qasm, shots=1)


def test_gate_qubit_reg(stdgates):
    qasm = """
    include "stdgates.inc";
    qubit[3] qs;
    qubit q;

    x qs[{0, 2}];
    ctrl @ rx(π/2) qs, q;
    """
    context = Interpreter().run(qasm, shots=0)
    assert np.allclose(
        context.quantum_simulation.state_vector,
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
            -1 / np.sqrt(2),
            -1j / np.sqrt(2),
            0,
            0,
            0,
            0,
        ],
    )


def test_gate_qubit_reg_size_mismatch(stdgates):
    qasm = """
    include "stdgates.inc";
    qubit[3] qs;
    qubit q;

    x qs[{0, 2}];
    ctrl @ rx(π/2) qs, qs[0:1];
    """
    size_mismatch = "Qubit registers must all be the same length."
    with pytest.raises(ValueError, match=size_mismatch):
        Interpreter().run(qasm, shots=0)


def test_gate_qubit_reg_shots(stdgates):
    qasm = """
    include "stdgates.inc";
    output bit[4] x;

    qubit[3] qs;
    qubit q;

    h q;
    cx q, qs;

    x[0] = measure q;
    x[1:] = measure qs;
    """
    context = Interpreter().run(qasm, shots=10)
    for outcome in context.shot_data["x"]:
        assert outcome in ("0000", "1111")


def test_pragma():
    qasm = """
    qubit q;
    #pragma {"braket result state_vector";}
    """
    Interpreter().run(qasm, shots=0)


def test_subroutine():
    qasm = """
    const int[8] n = 4;
    def parity(bit[n] cin) -> bit {
      bit c = false;
      for i in [0: n - 1] {
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


def test_void_subroutine(stdgates):
    qasm = """
    include "stdgates.inc";

    def flip(qubit q) {
      x q;
    }

    qubit[2] qs;
    flip(qs[0]);
    """
    context = Interpreter().run(qasm)
    assert np.allclose(
        context.quantum_simulation.state_vector,
        [0, 0, 1, 0],
    )


def test_array_ref_subroutine(stdgates):
    qasm = """
    include "stdgates.inc";
    output int[16] total_1;
    output int[16] total_2;

    def sum(const array[int[8], #dim = 1] arr) -> int[16] {
        int[16] size = sizeof(arr);
        int[16] x = 0;
        for i in [0:size - 1] {
            x += arr[i];
        }
        return x;
    }
    
    array[int[8], 5] array_1 = {1, 2, 3, 4, 5};
    array[int[8], 10] array_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    total_1 = sum(array_1);
    total_2 = sum(array_2);
    """
    context = Interpreter().run(qasm, shots=1)
    assert shot_data_is_equal(
        context.shot_data,
        {
            "total_1": [15],
            "total_2": [55],
        },
    )


def test_subroutine_array_reference_mutation(stdgates):
    qasm = """
    include "stdgates.inc";

    def mutate_array(mutable array[int[8], #dim = 1] arr) {
        int[16] size = sizeof(arr);
        for i in [0:size - 1] {
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


def test_subroutine_array_reference_const_mutation(stdgates):
    qasm = """
    include "stdgates.inc";

    def mutate_array(const array[int[8], #dim = 1] arr) {
        int[16] size = sizeof(arr);
        for i in [0:size - 1] {
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
