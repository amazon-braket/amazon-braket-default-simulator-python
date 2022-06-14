import re

import numpy as np
import pytest
from openqasm3.ast import (
    ArrayLiteral,
    ArrayType,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    FloatLiteral,
    FloatType,
    Identifier,
    IndexedIdentifier,
    IntegerLiteral,
    IntType,
    QuantumGate,
    QuantumGateDefinition,
    UintType,
)

from braket.default_simulator import StateVectorSimulation
from braket.default_simulator.gate_operations import U
from braket.default_simulator.openqasm import data_manipulation
from braket.default_simulator.openqasm.circuit import Circuit
from braket.default_simulator.openqasm.data_manipulation import (
    convert_bool_array_to_string,
    string_to_bin,
)
from braket.default_simulator.openqasm.interpreter import Interpreter
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
    assert context.get_value("pos") == FloatLiteral(10)
    assert context.get_value("neg") == FloatLiteral(-4)
    assert context.get_value("precise") == FloatLiteral(np.pi)


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

    assert context.get_value("copy_bit") == data_manipulation.convert_string_to_bool_array(
        BitstringLiteral(0b_10001000, 8)
    )
    assert context.get_value("copy_int") == IntegerLiteral(100)
    assert context.get_value("copy_uint") == IntegerLiteral(8)
    # notice the reduced precision compared to np.pi from float[16]
    assert context.get_value("copy_float") == FloatLiteral(3.140625)


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
        BitstringLiteral(0b_1111, 4)
    )
    assert context.get_value("one_b") == data_manipulation.convert_string_to_bool_array(
        BitstringLiteral(0b_0001, 4)
    )
    assert context.get_value("trunc_b") == data_manipulation.convert_string_to_bool_array(
        BitstringLiteral(0b_000, 3)
    )
    assert context.get_value("neg_fifteen_b") == data_manipulation.convert_string_to_bool_array(
        BitstringLiteral(0b_10001, 5)
    )
    assert context.get_value("neg_one_b") == data_manipulation.convert_string_to_bool_array(
        BitstringLiteral(0b_11111, 5)
    )
    assert context.get_value("bit_slice") == data_manipulation.convert_string_to_bool_array(
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
    int[8] y = 0;
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
    assert context.get_value("y") == IntegerLiteral(sum((2, 4, 6)))


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
                    FloatLiteral(np.pi),
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
                    FloatLiteral(np.pi),
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
                    FloatLiteral(np.pi),
                    IntegerLiteral(0),
                    FloatLiteral(np.pi),
                ],
                qubits=[Identifier("b")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    FloatLiteral(np.pi),
                    IntegerLiteral(0),
                    Identifier("p"),
                ],
                qubits=[Identifier("a")],
            ),
            QuantumGate(
                modifiers=[],
                name=Identifier("U"),
                arguments=[
                    FloatLiteral(np.pi),
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

    U(π, 0, my_pi) q1;
    x q2;
    x2(my_pi) qs[0];
    """
    circuit = Interpreter().build_circuit(qasm)
    expected_circuit = Circuit(
        instructions=[
            U((0,), np.pi, 0, np.pi, ()),
            U((1,), np.pi, 0, np.pi, ()),
            U((2,), np.pi, 0, np.pi, ()),
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
    """
    circuit = Interpreter().build_circuit(qasm)
    collapsed = np.linalg.multi_dot([instruction.matrix for instruction in circuit.instructions])
    assert np.allclose(collapsed, np.eye(2**circuit.num_qubits))


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


def test_measurement():
    qasm = """
    qubit q;
    measure q;
    """
    cannot_measure = "Measurement not supported"
    with pytest.raises(NotImplementedError, match=cannot_measure):
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
    print(simulation.state_vector)
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


def test_adder(adder):
    circuit = Interpreter().build_circuit("adder.qasm", {"a_in": 1, "b_in": 15}, is_file=True)
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
    assert context.get_value("small_pi") == FloatLiteral(3.140625)
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
    int[8] doubled;

    doubled = in_int * 2;
    """
    context = Interpreter().run(qasm, inputs={"in_int": in_int})
    assert context.get_value("doubled") == IntegerLiteral(in_int * 2)


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
    """
    missing_input = "Missing input variable 'in_int'."
    with pytest.raises(NameError, match=missing_input):
        Interpreter().run(qasm)


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
    invalid_float = "Float size must be one of {16, 32, 64, 128}."
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


def test_gate_qubit_reg(stdgates):
    qasm = """
    include "stdgates.inc";
    qubit[3] qs;
    qubit q;

    x qs[{0, 2}];
    ctrl @ rx(π/2) qs, q;
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
        Interpreter().run(qasm)


def test_pragma():
    qasm = f"""
    qubit q;
    #pragma {{"{string_to_bin("braket result state_vector")}";}}
    """
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


def test_void_subroutine(stdgates):
    qasm = """
    include "stdgates.inc";

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


def test_array_ref_subroutine(stdgates):
    qasm = """
    include "stdgates.inc";
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


def test_subroutine_array_reference_mutation(stdgates):
    qasm = """
    include "stdgates.inc";

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


def test_subroutine_array_reference_const_mutation(stdgates):
    qasm = """
    include "stdgates.inc";

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
