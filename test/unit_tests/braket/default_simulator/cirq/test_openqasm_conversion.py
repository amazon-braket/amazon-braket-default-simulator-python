import cirq
import numpy as np
import pytest

from braket.default_simulator.cirq.program_context import ProgramContext
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator import StateVectorSimulation


@pytest.mark.parametrize(
    "openqasm",
    (
        (
            """
        OPENQASM 3.0;
        bit[1] b;
        qubit[1] q;
        h q[0];
        b[0] = measure q[0];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        b[0] = measure q[0];
        b[1] = measure q[1];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        cnot q[0], q[1];
        b[0] = measure q[0];
        b[1] = measure q[1];
        """
        ),
        (
            """
            OPENQASM 3.0;
            bit[2] b;
            qubit[2] q;
            pow(0.5) @ cnot q[0], q[1];
            b[0] = measure q[0];
            b[1] = measure q[1];
            """
        ),
        (
            """
            OPENQASM 3.0;
            bit[2] b;
            qubit[2] q;
            ctrl @ x q[0], q[1];
            b[0] = measure q[0];
            b[1] = measure q[1];
            """
        ),
        (
            """
        OPENQASM 3.0;
        bit[1] b;
        qubit[1] q;
        rx(0.15) q[0];
        b[0] = measure q[0];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[1] b;
        qubit[1] q;
        ry(0.2) q[0];
        b[0] = measure q[0];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[1] b;
        qubit[1] q;
        rz(0.2) q[0];
        b[0] = measure q[0];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        x q[0];
        x q[1];
        b[0] = measure q[0];
        b[1] = measure q[1];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[3] b;
        qubit[3] q;
        y q[0];
        y q[1];
        y q[2];
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[1] b;
        qubit[1] q;
        s q[0];
        b[0] = measure q[0];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[1] b;
        qubit[1] q;
        t q[0];
        b[0] = measure q[0];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        cphaseshift(0.15) q[1], q[0];
        b[0] = measure q[0];
        b[1] = measure q[1];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        swap q[0], q[1];
        b[0] = measure q[0];
        b[1] = measure q[1];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[3] b;
        qubit[3] q;
        ccnot q[0], q[1], q[2];
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """
        ),
        (
            """
        OPENQASM 3.0;
        bit[3] b;
        qubit[3] q;
        cswap q[0], q[1], q[2];
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """
        ),
        (
            """
            OPENQASM 3.0;
            bit[5] b;
            qubit[5] q;
            rx(0.15) q[0];
            ry(0.2) q[1];
            rz(0.25) q[2];
            h q[3];
            cnot q[0], q[2];
            cnot q[1], q[3];
            x q[1];
            x q[3];
            y q[1];
            y q[3];
            z q[1];
            z q[2];
            s q[2];
            t q[1];
            cphaseshift(0.15) q[2], q[4];
            swap q[3], q[4];
            ccnot q[2], q[3], q[4];
            cswap q[1], q[3], q[4];
            b[0] = measure q[0];
            b[1] = measure q[1];
            b[2] = measure q[2];
            b[3] = measure q[3];
            b[4] = measure q[4];
            """
        ),
    ),
)
def test_openqasm_conversion_to_cirq(openqasm):
    circuit = Interpreter(ProgramContext()).build_circuit(
        source=openqasm,
    )

    braket_circuit = Interpreter().build_circuit(
        source=openqasm,
    )
    simulation = StateVectorSimulation(braket_circuit.num_qubits, 1, 1)
    simulation.evolve(braket_circuit.instructions)

    assert isinstance(circuit, cirq.Circuit)
    assert np.allclose(cirq.final_state_vector(circuit), simulation.state_vector)
