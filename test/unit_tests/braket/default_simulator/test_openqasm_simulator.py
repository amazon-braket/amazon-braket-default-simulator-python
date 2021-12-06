import re

import numpy as np
import pytest

from braket.default_simulator.openqasm_helpers import BitVariable, QubitPointer, IntVariable, UintVariable, \
    FloatVariable
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
    "size", ("2 + 4", "2**3")
)
def test_qubit_expression_declaration(size):
    qasm = f"""
    qubit[{size}] a;
    """
    simulator = QasmSimulator()
    with pytest.raises(NotImplementedError):
        simulator.run_qasm(qasm)


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

