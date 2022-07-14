import numpy as np
from braket.ir.openqasm import Program

from braket.default_simulator.openqasm_density_matrix_simulator import (
    OpenQASMDensityMatrixSimulator,
)


def test_openqasm_density_matrix_simulator():
    noisy_bell_qasm = """
    qubit[2] qs;

    h qs[0];
    cnot qs[0], qs[1];

    #pragma braket noise bit_flip(.2) qs[1]

    #pragma braket result probability
    """
    device = OpenQASMDensityMatrixSimulator()
    program = Program(source=noisy_bell_qasm)
    result = device.run(program)
    probabilities = result.resultTypes[0].value
    assert np.allclose(probabilities, [0.4, 0.1, 0.1, 0.4])
