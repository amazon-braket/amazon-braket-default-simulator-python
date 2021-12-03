import numpy as np
import pytest

from braket.default_simulator.openqasm_helpers import BitVariable, QubitPointer
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


@pytest.mark.parametrize(
    "size", (-1, "2 + 4", "2**3")
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
    bit single_initialized_float = π;
    bit[2] register_uninitialized;
    bit[2] register_initialized = "01";
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.qasm_variables == {
        "single_uninitialized": BitVariable("single_uninitialized"),
        "single_initialized_int": BitVariable("single_initialized_int", 0),
        "single_initialized_bool": BitVariable("single_initialized_bool", 1),
        "single_initialized_float": BitVariable("single_initialized_float", 1),
        "register_uninitialized": BitVariable("register_uninitialized", size=2),
        "register_initialized": BitVariable("register_initialized", "01", 2),
    }


def test_int_declaration():
    qasm = """
        // int[1] b;
        // float[32] f = π;
        int[8] b1 = 1;
        int[16] bs;
        bit[2] b01 = "01";
        bit[3] b3 = 3;
        """
    # print()
    # simulator = QasmSimulator()
    # simulator.run_qasm(qasm)
