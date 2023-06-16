import cirq
import numpy as np
import pytest

from braket.default_simulator import DensityMatrixSimulation
from braket.default_simulator.cirq.cirq_program_context import CirqProgramContext
from braket.default_simulator.openqasm.interpreter import Interpreter


@pytest.mark.parametrize(
    "openqasm",
    (
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            x q[0];
            #pragma braket noise bit_flip(0.1) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise generalized_amplitude_damping(0.1, 0.1) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise phase_flip(0.2) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise depolarizing(0.5) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise depolarizing(0.5) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise amplitude_damping(0.8) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise amplitude_damping(0.8) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise generalized_amplitude_damping(0.1, 0.3) q[0]
            b[0] = measure q[0];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            #pragma braket noise phase_damping(0.1) q[0]
            b[0] = measure q[0];
            """
        ),
    ),
)
def test_openqasm_conversion_basic_gates_to_cirq(openqasm):
    cirq_circuit = Interpreter(CirqProgramContext()).build_circuit(
        source=openqasm,
    )

    braket_circuit = Interpreter().build_circuit(
        source=openqasm,
    )
    simulation = DensityMatrixSimulation(braket_circuit.num_qubits, 1)
    simulation.evolve(braket_circuit.instructions)

    assert isinstance(cirq_circuit, cirq.Circuit)
    assert np.allclose(cirq.final_density_matrix(cirq_circuit), simulation.density_matrix)
