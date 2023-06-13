import cirq
import pytest

from braket.default_simulator.cirq.program_context import ProgramContext
from braket.default_simulator.openqasm.interpreter import Interpreter


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
        bit[12] b;
        qubit[12] q;
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
        cphaseshift(0.15) q[4], q[2];
        swap q[4], q[5];
        ccnot q[6], q[7], q[8];
        cswap q[9], q[10], q[11];
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        b[3] = measure q[3];
        b[4] = measure q[4];
        b[5] = measure q[5];
        b[6] = measure q[6];
        b[7] = measure q[7];
        b[8] = measure q[8];
        b[9] = measure q[9];
        b[10] = measure q[10];
        b[11] = measure q[11];
        """
        ),
    ),
)
def test_openqasm_conversion_to_cirq(openqasm):
    circuit = Interpreter(ProgramContext()).build_circuit(
        source=openqasm,
    )

    assert isinstance(circuit, cirq.Circuit)
