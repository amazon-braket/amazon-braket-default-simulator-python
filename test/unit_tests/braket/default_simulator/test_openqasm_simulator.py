import numpy as np

from braket.default_simulator.openqasm_simulator import QasmSimulator, QasmVariable


def test_qubit_declaration():
    qasm = """
    qubit q;
    qubit[0] p;
    qubit[4] a;
    """
    simulator = QasmSimulator()
    simulator.run_qasm(qasm)

    assert simulator.variables == {
        "q": QasmVariable("q", 0, True),
        "p": QasmVariable("p", slice(1, 1), True),
        "a": QasmVariable("a", slice(1, 5), True),
    }
    assert simulator.get_variable_value("q") == 0
    assert np.array_equal(simulator.get_variable_value("p"), [])
    assert np.array_equal(simulator.get_variable_value("a"), [0, 0, 0, 0])