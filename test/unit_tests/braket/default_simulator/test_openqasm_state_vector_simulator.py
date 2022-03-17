import numpy as np
import pytest
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program


@pytest.fixture
def ghz(pytester, stdgates):
    pytester.makefile(
        ".qasm",
        ghz="""
            include "stdgates.inc";

            qubit[3] q;
            bit[3] c;

            h q[0];
            cx q[0], q[1];
            cx q[1], q[2];

            c = measure q;
        """,
    )


@pytest.fixture
def adder(pytester, stdgates):
    pytester.makefile(
        ".qasm",
        adder="""
            /*
             * quantum ripple-carry adder
             * Cuccaro et al, quant-ph/0410184
             * (adjusted for Braket)
             */
            OPENQASM 3;
            include "stdgates.inc";

            input uint[4] a_in;
            input uint[4] b_in;
            output bit[5] ans;

            gate majority a, b, c {
                cx c, b;
                cx c, a;
                ccx a, b, c;
            }

            gate unmaj a, b, c {
                ccx a, b, c;
                cx c, a;
                cx a, b;
            }

            qubit cin;
            qubit[4] a;
            qubit[4] b;
            qubit cout;

            // initialize qubits
            reset cin;
            reset a;
            reset b;
            reset cout;

            // set input states
            for i in [0: 3] {
              if(bool(a_in[i])) x a[i];
              if(bool(b_in[i])) x b[i];
            }

            // add a to b, storing result in b
            majority cin, b[3], a[3];
            for i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
            cx a[0], cout;
            for i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
            unmaj cin, b[3], a[3];

            // measure results
            ans[0] = measure cout;
            ans[1:4] = measure b[0:3];
        """,
    )


@pytest.mark.parametrize("shots", (1, 2, 10))
def test_ghz(ghz, shots):
    device = LocalSimulator("braket_oq3_sv")
    program = Program(source="ghz.qasm")
    result = device.run(program, shots=shots).result()

    assert len(result.output_variables["c"]) == shots
    for outcome in result.output_variables["c"]:
        assert outcome in ("000", "111")


def test_adder(adder):
    for _ in range(3):
        device = LocalSimulator("braket_oq3_sv")
        a, b = np.random.randint(0, 16, 2)
        inputs = {"a_in": a, "b_in": b}
        program = Program(source="adder.qasm", inputs=inputs)
        result = device.run(program, shots=1).result()
        ans = int(result.output_variables["ans"][0], base=2)
        assert a + b == ans
