import numpy as np
import pytest
from openqasm.parser.antlr.qasm_parser import parse

from braket.default_simulator.openqasm_helpers import BitVariable, QubitPointer, IntVariable, UintVariable, \
    FloatVariable, AngleVariable, BoolVariable, ComplexVariable
from braket.default_simulator.openqasm_simulator import QasmSimulator


def test_qubit_declaration():
    qasm = """
    qubit q;
    qubit[4] qs;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "q": QubitPointer("q", 0),
        "qs": QubitPointer("qs", slice(1, 5), 4),
    }

    assert np.isnan(simulator.get_qubit_state("q"))

    assert np.all(np.isnan(simulator.get_qubit_state("qs")))
    assert simulator.get_qubit_state("qs").shape == (4,)
    assert simulator.qubits.shape == (5,)


@pytest.mark.parametrize(
    "size", (0, 4.3)
)
def test_qubit_bad_declaration(size):
    qasm = f"""
    qubit[{size}] qs;
    """
    bad_size = (
        f"Qubit register size must be a positive integer. "
        f"Provided size '{size}' for qubit register 'qs'."
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
        "Provided size '-1' for qubit register 'a'."
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
        "a": QubitPointer("a", slice(0, 6), size=6),
    }


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
        "single_uninitialized": BitVariable("single_uninitialized"),
        "single_initialized_int": BitVariable("single_initialized_int", 0),
        "single_initialized_bool": BitVariable("single_initialized_bool", 1),
        "register_uninitialized": BitVariable("register_uninitialized", size=2),
        "register_initialized": BitVariable("register_initialized", "01", 2),
    }


@pytest.mark.parametrize("size", (2.3, 0, -1))
def test_bit_declaration_bad_size(size):
    qasm = f"bit[{size}] bad_size;"
    bad_size = (
        f"Bit register size must be a positive integer. "
        f"Provided size '{size}' for bit register 'bad_size'."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


def test_bit_declaration_wrong_size():
    qasm = """
    bit[3] size_mismatch = "01";
    """
    wrong_size = (
        f"Invalid value to initialize bit register 'size_mismatch': '01'. "
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
        f"Invalid value to initialize bit variable 'wrong_type': {repr(value)}. "
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
        f"Invalid value to initialize bit register 'reg_wrong_type': {repr(value)}. "
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
        "Integer overflow for integer register 'pos_overflow'. "
        "Value '5' is outside the range for an integer register of size '3'."
    )
    neg_overflow = (
       "Integer overflow for integer register 'neg_overflow'. "
       "Value '-6' is outside the range for an integer register of size '3'."
    )
    with pytest.warns(UserWarning) as warn_info:
        simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": IntVariable("uninitialized", size=8),
        "pos": IntVariable("pos", 10, 8),
        "neg": IntVariable("neg", -4, 5),
        "pos_overflow": IntVariable("pos_overflow", 1, 3),
        "neg_overflow": IntVariable("neg_overflow", -2, 3),
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
        f"Provided size '{size}' for integer register 'bad_size'."
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
        f"Not a valid value for integer register 'wrong_type': {repr(value)}"
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
        "Integer overflow for unsigned integer register 'pos_overflow'. "
        "Value '8' is outside the range for an unsigned integer register of size '3'."
    )
    with pytest.warns(UserWarning, match=pos_overflow):
        simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": UintVariable("uninitialized", size=8),
        "pos": UintVariable("pos", 10, 8),
        "pos_not_overflow": UintVariable("pos_not_overflow", 5, 3),
        "pos_overflow": UintVariable("pos_overflow", 0, 3),
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
        "uninitialized": FloatVariable("uninitialized", size=16),
        "pos": FloatVariable("pos", 10, 32),
        "neg": FloatVariable("neg", -4, 64),
        "precise": FloatVariable("precise", np.pi, 128),
    }


def test_float_declaration_bad_value():
    qasm = """
    float[16] string = "not a float";
    """
    bad_value = (
        "Not a valid value for float 'string': 'not a float'"
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
        f"Provided size '{size}' for float 'bad_size'."
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
        "uninitialized": AngleVariable("uninitialized", size=20),
        "initialized": AngleVariable("initialized", three_pi_over_two, 20),
        "multiples_of_pi_are_compact": AngleVariable(
            "multiples_of_pi_are_compact", three_pi_over_two, 2
        ),
        "integers_are_harder": AngleVariable("integers_are_harder", 3, 5),
    }

    # note how the multiple of pi is preserved exactly while the integer loses precision
    assert simulator.qasm_variables["multiples_of_pi_are_compact"].value == three_pi_over_two
    assert simulator.qasm_variables["integers_are_harder"].value == 2.945243112740431


@pytest.mark.parametrize("size", (2.3, 0, -1))
def test_angle_declaration_bad_size(size):
    qasm = f"angle[{size}] bad_size;"
    bad_size = (
        f"Angle size must be a positive integer. "
        f"Provided size '{size}' for angle 'bad_size'."
    )
    simulator = QasmSimulator()
    with pytest.raises(ValueError, match=bad_size):
        simulator.run_qasm(qasm)


def test_angle_declaration_wrong_type():
    qasm = f"""
    angle[8] wrong_type = "a string";
    """
    wrong_type = (
        f"Not a valid value for angle 'wrong_type': 'a string'"
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
        "uninitialized": BoolVariable("uninitialized", None),
        "t_bool": BoolVariable("t_bool", True),
        "t_int": BoolVariable("t_int", True),
        "t_float": BoolVariable("t_float", True),
        "f_bool": BoolVariable("f_bool", False),
        "f_int": BoolVariable("f_int", False),
        "f_float": BoolVariable("f_float", False),
    }


def test_complex_declaration():
    qasm = """
    complex[int[8]] real = 1 + 2im;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "uninitialized": ComplexVariable("uninitialized", None),
        "real": ComplexVariable("real", 1),
        "t_int": BoolVariable("t_int", True),
        "t_float": BoolVariable("t_float", True),
        "f_bool": BoolVariable("f_bool", False),
        "f_int": BoolVariable("f_int", False),
        "f_float": BoolVariable("f_float", False),
    }


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
        "int_var": IntVariable("int_var", -2, 16),
        "float_var": FloatVariable("float_var", 2.0, 16),
        "bit_var": BitVariable("bit_var", "10", 2),
        "uint_var": UintVariable("uint_var", 7, 5),
        "angle_var": AngleVariable("angle_var", np.pi, 10),
        "bool_var": BoolVariable("bool_var", False),
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
        "int_var": IntVariable("int_var", -2, 16),
        "float_var": FloatVariable("float_var", 2.0, 16),
        "bit_var": BitVariable("bit_var", "10", 2),
        "uint_var": UintVariable("uint_var", 7, 5),
        "angle_var": AngleVariable("angle_var", np.pi, 10),
        "bool_var": BoolVariable("bool_var", False),
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
        "x": IntVariable("x", 1 if conditional == "true" else 2, 16),
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
        "x": IntVariable("x", 1 if conditional == "true" else 2, 16),
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
        "locally_overridden": IntVariable("locally_overridden", 1, 16),
        "globally_changed": IntVariable("globally_changed", 2, 16),
    }


def test_array():
    qasm = f"""
        array[int[32], 5] myArray = {{0, 1, 2, 3, 4}};
        """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "locally_overridden": IntVariable("locally_overridden", 1, 16),
        "globally_changed": IntVariable("globally_changed", 2, 16),
    }
