import re

import numpy as np
import pytest

from braket.default_simulator.openqasm_helpers import Bit, QubitPointer, Int, Uint, \
    Float, Angle, Bool, Complex, Array, sample_qubit
from braket.default_simulator.openqasm_simulator import QasmSimulator


def test_qubit_declaration():
    qasm = """
    qubit q;
    qubit[4] qs;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "q": QubitPointer(0),
        "qs": QubitPointer(slice(1, 5), 4),
    }

    assert np.all(np.isnan(simulator.get_qubit_state("q")))
    assert np.all(np.isnan(simulator.get_qubit_state("qs")))
    assert simulator.get_qubit_state("qs").shape == (4, 2)
    assert simulator.qubits.shape == (5, 2)


@pytest.mark.parametrize(
    "size", (0, 4.3)
)
def test_qubit_bad_declaration(size):
    qasm = f"""
    qubit[{size}] qs;
    """
    bad_size = (
        f"Qubit register size must be a positive integer. "
        f"Provided size '{size}' for qubit register."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


def test_qubit_negative_size_declaration():
    qasm = f"""
    qubit[-1] a;
    """
    size_must_be_pos = (
        "Qubit register size must be a positive integer. "
        "Provided size '-1' for qubit register."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=size_must_be_pos):
        simulator.run_qasm(qasm)


@pytest.mark.parametrize(
    "size", ("2 + 4", "2 * 3")
)
def test_qubit_expression_declaration(size):
    qasm = f"""
    qubit[{size}] a;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "a": QubitPointer(slice(0, 6), size=6),
    }


def test_qubit_reset():
    qasm = """
    qubit q;
    qubit[4] qs;
    
    reset q;
    reset qs;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "q": QubitPointer(0),
        "qs": QubitPointer(slice(1, 5), 4),
    }

    assert np.all(simulator.get_qubit_state("q") == [[1, 0]])
    assert np.all(simulator.get_qubit_state("qs") == [
        [1, 0], [1, 0], [1, 0], [1, 0]
    ])


def test_qubit_measure():
    qasm = """
    qubit q;
    qubit[4] qs;
    
    bit b;
    bit[4] bs;
    
    reset q;
    reset qs;
    
    b = measure q;
    bs = measure qs;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "q": QubitPointer(0),
        "qs": QubitPointer(slice(1, 5), 4),
        "b": Bit(0),
        "bs": Bit("0000", 4),
    }

    assert np.all(simulator.get_qubit_state("q") == [[1, 0]])
    assert np.all(simulator.get_qubit_state("qs") == [
        [1, 0], [1, 0], [1, 0], [1, 0]
    ])


def test_bit_declaration():
    qasm = """
    bit single_uninitialized;
    bit single_initialized_int = 0;
    bit single_initialized_bool = true;
    bit[2] register_uninitialized;
    bit[2] register_initialized = "01";
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "single_uninitialized": Bit(),
        "single_initialized_int": Bit(0),
        "single_initialized_bool": Bit(1),
        "register_uninitialized": Bit(size=2),
        "register_initialized": Bit("01", 2),
    }


@pytest.mark.parametrize("size", (2.3, 0, -1))
def test_bit_declaration_bad_size(size):
    qasm = f"bit[{size}] bad_size;"
    bad_size = (
        f"Bit register size must be a positive integer. "
        f"Provided size '{size}' for bit register."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


def test_bit_declaration_wrong_size():
    qasm = """
    bit[3] size_mismatch = "01";
    """
    wrong_size = (
        f"Invalid value to initialize bit register: '01'. "
        "Provided value must be a binary string of length equal to "
        f"given size, 3."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=wrong_size):
        simulator.run_qasm(qasm)


@pytest.mark.parametrize("value", ("1", 2))
def test_bit_declaration_wrong_type(value):
    qasm_value = f'"{value}"' if isinstance(value, str) else value
    qasm = f"""
    bit wrong_type = {qasm_value};
    """
    wrong_type = (
        f"Invalid value to initialize bit variable: {repr(value)}. "
        "Provided value must be a boolean value."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=wrong_type):
        simulator.run_qasm(qasm)


@pytest.mark.parametrize("value", ("24", 3))
def test_bit_declaration_wrong_type_register(value):
    qasm_value = f'"{value}"' if isinstance(value, str) else value
    qasm = f"""
    bit[2] reg_wrong_type = {qasm_value};
    """
    wrong_type = (
        f"Invalid value to initialize bit register: {repr(value)}. "
        "Provided value must be a binary string of length equal to "
        f"given size, 2."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=wrong_type):
        simulator.run_qasm(qasm)


def test_int_declaration():
    qasm = """
    int[8] uninitialized;
    int[8] pos = 10;
    int[5] neg = -4;
    int[3] pos_overflow = 5;
    int[3] neg_overflow = -6;
    """
    simulator = QasmSimulator()
    pos_overflow = (
        "Integer overflow for integer register. "
        "Value '5' is outside the range for an integer register of size '3'."
    )
    neg_overflow = (
        "Integer overflow for integer register. "
        "Value '-6' is outside the range for an integer register of size '3'."
    )
    with pytest.warns(UserWarning) as warn_info:
        simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": Int(size=8),
        "pos": Int(10, 8),
        "neg": Int(-4, 5),
        "pos_overflow": Int(1, 3),
        "neg_overflow": Int(-2, 3),
    }

    warnings = {(warn.category, warn.message.args[0]) for warn in warn_info}
    assert warnings == {
        (UserWarning, pos_overflow),
        (UserWarning, neg_overflow),
    }


@pytest.mark.parametrize("size", (2.3, 0, -1))
def test_int_declaration_bad_size(size):
    qasm = f"int[{size}] bad_size;"
    bad_size = (
        f"Integer register size must be a positive integer. "
        f"Provided size '{size}' for integer register."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


@pytest.mark.parametrize("value", ("2", 2.3))
def test_int_declaration_wrong_type(value):
    qasm_value = f'"{value}"' if isinstance(value, str) else value
    qasm = f"""
    int[8] wrong_type = {qasm_value};
    """
    wrong_type = (
        f"Not a valid value for integer register: {repr(value)}"
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=wrong_type):
        simulator.run_qasm(qasm)


def test_uint_declaration():
    qasm = """
    uint[8] uninitialized;
    uint[8] pos = 10;
    uint[3] pos_not_overflow = 5;
    uint[3] pos_overflow = 8;
    """
    simulator = QasmSimulator()
    pos_overflow = (
        "Integer overflow for unsigned integer register. "
        "Value '8' is outside the range for an unsigned integer register of size '3'."
    )
    with pytest.warns(UserWarning, match=pos_overflow):
        simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": Uint(size=8),
        "pos": Uint(10, 8),
        "pos_not_overflow": Uint(5, 3),
        "pos_overflow": Uint(0, 3),
    }


def test_float_declaration():
    qasm = """
    float[16] uninitialized;
    float[32] pos = 10;
    float[64] neg = -4.;
    float[128] precise = π;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": Float(size=16),
        "pos": Float(10, 32),
        "neg": Float(-4, 64),
        "precise": Float(np.pi, 128),
    }


def test_float_declaration_bad_value():
    qasm = """
    float[16] string = "not a float";
    """
    bad_value = re.escape(
        "Not a valid value for float[16]: 'not a float'"
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_value):
        simulator.run_qasm(qasm)


@pytest.mark.parametrize("size", (-1, 0, .3, 8))
def test_float_declaration_bad_size(size):
    qasm = f"""
    float[{size}] bad_size;
    """
    bad_value = (
        "Float size must be one of {16, 32, 64, 128}. "
        f"Provided size '{size}' for float."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_value):
        simulator.run_qasm(qasm)


def test_angle_declaration():
    qasm = """
    angle[20] uninitialized;
    angle[20] initialized = 3*π/2;
    angle[2] multiples_of_pi_are_compact = 3*π/2;
    angle[5] integers_are_harder = 3;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    three_pi_over_two = 4.71238898038469

    assert simulator.qasm_variables == {
        "uninitialized": Angle(size=20),
        "initialized": Angle(three_pi_over_two, 20),
        "multiples_of_pi_are_compact": Angle(three_pi_over_two, 2),
        "integers_are_harder": Angle(3, 5),
    }

    # note how the multiple of pi is preserved exactly while the integer loses precision
    assert simulator.qasm_variables["multiples_of_pi_are_compact"].value == three_pi_over_two
    assert simulator.qasm_variables["integers_are_harder"].value == 2.945243112740431


@pytest.mark.parametrize("size", (2.3, 0, -1))
def test_angle_declaration_bad_size(size):
    qasm = f"angle[{size}] bad_size;"
    bad_size = (
        f"Angle size must be a positive integer. "
        f"Provided size '{size}' for angle."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


def test_angle_declaration_wrong_type():
    qasm = f"""
    angle[8] wrong_type = "a string";
    """
    wrong_type = (
        f"Not a valid value for angle: 'a string'"
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=wrong_type):
        simulator.run_qasm(qasm)


def test_bool_declaration():
    qasm = """
    bool uninitialized;
    bool t_bool = true;
    bool t_int = 10;
    bool t_float = -π;
    bool f_bool = false;
    bool f_int = 0;
    bool f_float = 0.0;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": Bool(None),
        "t_bool": Bool(True),
        "t_int": Bool(True),
        "t_float": Bool(True),
        "f_bool": Bool(False),
        "f_int": Bool(False),
        "f_float": Bool(False),
    }


# def test_complex_declaration():
#     qasm = """
#     complex[int[8]] uninitialized;
#     # complex[int[8]] comp = 1 + 2im;
#     """
#     simulator = QasmSimulator()
#     simulator.run_qasm(qasm)
#
#     assert simulator.qasm_variables == {
#         "uninitialized": Complex(None, Int(size=8)),
#         "comp": Complex((Int(1, 8), Int(2, 8)), Int(size=8)),
#     }


def test_assign_declared():
    qasm = """
    int[16] int_var;
    float[16] float_var;
    bit[2] bit_var;
    uint[5] uint_var;
    angle[10] angle_var;
    bool bool_var;
    int_var = -2;
    float_var = 2;
    bit_var = "10";
    uint_var = 7;
    angle_var = π;
    bool_var = false;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "int_var": Int(-2, 16),
        "float_var": Float(2.0, 16),
        "bit_var": Bit("10", 2),
        "uint_var": Uint(7, 5),
        "angle_var": Angle(np.pi, 10),
        "bool_var": Bool(False),
    }


def test_assign_instantiated():
    qasm = """
    int[16] int_var = 1;
    float[16] float_var = 1.0;
    bit[2] bit_var = "00";
    uint[5] uint_var = 3;
    angle[10] angle_var = π / 3;
    bool bool_var = true;
    int_var = -2;
    float_var = 2;
    bit_var = "10";
    uint_var = 7;
    angle_var = π;
    bool_var = false;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "int_var": Int(-2, 16),
        "float_var": Float(2.0, 16),
        "bit_var": Bit("10", 2),
        "uint_var": Uint(7, 5),
        "angle_var": Angle(np.pi, 10),
        "bool_var": Bool(False),
    }


def test_assign_undeclared():
    qasm = """
    x = 2;
    """
    not_in_scope = (
        "Variable 'x' not in scope."
    )
    simulator = QasmSimulator()
    with pytest.raises(NameError, match=not_in_scope):
        simulator.run_qasm(qasm)


@pytest.mark.parametrize(
    "conditional", ("true", "false")
)
def test_if(conditional):
    qasm = f"""
    int[16] x = 2;
    if ({conditional}) {{
        x = 1;
    }}
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "x": Int(1 if conditional == "true" else 2, 16),
    }


@pytest.mark.parametrize(
    "conditional", ("true", "false")
)
def test_if_else(conditional):
    qasm = f"""
    int[16] x;
    if ({conditional}) {{
        x = 1;
    }} else {{
        x = 2;
    }}
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "x": Int(1 if conditional == "true" else 2, 16),
    }


def test_if_scope():
    qasm = f"""
        int[16] locally_overridden = 1;
        int[16] globally_changed = 1;
        if (true) {{
            int[16] locally_overridden = 2;
            globally_changed = locally_overridden;
        }}
        """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "locally_overridden": Int(1, 16),
        "globally_changed": Int(2, 16),
    }


def test_array():
    qasm = """
        array[int[32], 5] my_1d_array = {0, 1, 2, 3, 4};
        array[int[32], 2, 3] my_2d_array = {{0, 1, 2}, {3, 4, 5}};
        array[float[32], 1, 1, 1, 1, 1] my_5d_array = {{{{{π}}}}};
        array[float[32], 2] spunky_array = {0 + 1, -π};
        array[float[32], 2, 2] uninitialized_array;
        """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "my_1d_array": Array(
            [
                Int(0, 32),
                Int(1, 32),
                Int(2, 32),
                Int(3, 32),
                Int(4, 32),
            ],
            [5],
        ),
        "my_2d_array": Array(
            [
                [
                    Int(0, 32),
                    Int(1, 32),
                    Int(2, 32),
                ],
                [
                    Int(3, 32),
                    Int(4, 32),
                    Int(5, 32),
                ],
            ],
            [2, 3],
        ),
        "my_5d_array": Array([[[[[Float(np.pi, 32)]]]]], [1, 1, 1, 1, 1]),
        "spunky_array": Array(
            [
                Float(1, 32),
                Float(-np.pi, 32),
            ],
            [2],
        ),
        "uninitialized_array": Array(None, [2, 2]),
    }


def test_bad_size_array():
    qasm = """
    array[int[32], 1, 2.1] my_array;
    """
    bad_size = re.escape(
        "Integer register array dimensions must be positive integers. "
        "Provided dimensions '[1, 2.1]' for integer register array."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


def test_sample_qubit():
    assert sample_qubit([1, 0]) == 0
    assert sample_qubit([0, 1]) == 1


def test_gate_definition():
    qasm = """
    gate h_p(p) q {
       U(π/2, 0, p*π) q;
    }
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "q": QubitPointer(0),
        "qs": QubitPointer(slice(1, 5), 4),
    }

    # assert np.all(simulator.get_qubit_state("q") == [[1, 0]])
    # assert np.all(simulator.get_qubit_state("qs") == [
    #     [1, 0], [1, 0], [1, 0], [1, 0]
    # ])


def test_built_in_gate():
    qasm = """
    qubit[2] q;
    U(π/2, 0, π) q;
    misc(param) p, q, r;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "q": QubitPointer(0),
        "qs": QubitPointer(slice(1, 5), 4),
    }

    assert np.all(simulator.get_qubit_state("q") == [[1, 0]])
    assert np.all(simulator.get_qubit_state("qs") == [
        [1, 0], [1, 0], [1, 0], [1, 0]
    ])


def test_custom_gate():
    qasm = """
    gate h_p(p_loc) q_loc {
       U(π/2, 0, p_loc*π) q_loc;
    }
    
    qubit[2] qs;
    qubit decoy;
    qubit q;
    int[8] p = 1;
    
    reset qs;
    reset decoy;
    reset q;
    h_p(p) qs;
    h_p(p) q;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert np.all(np.isclose(
        simulator.qubits,
        [[0.70710678, 0.70710678],
         [0.70710678, 0.70710678],
         [1,          0         ],
         [0.70710678, 0.70710678]]
    ))