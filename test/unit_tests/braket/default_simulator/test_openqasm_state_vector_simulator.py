import numpy as np
import pytest
from braket.ir.openqasm import Program

from braket.default_simulator.openqasm_state_vector_simulator import OpenQASMStateVectorSimulator


def string_to_bin(string):
    """workaround for unsupported pragmas"""
    return "".join(np.binary_repr(ord(x), 8) for x in string)


def test_gphase():
    qasm = f"""
    qubit[2] qs;

    int[8] two = 2;

    gate x a {{ U(π, 0, π) a; }}
    gate cx c, a {{ ctrl @ x c, a; }}
    gate phase c, a {{
        gphase(π/2);
        pow(1) @ ctrl(two) @ gphase(π) c, a;
    }}
    gate h a {{ U(π/2, 0, π) a; }}

    inv @ U(π/2, 0, π) qs[0];
    cx qs[0], qs[1];
    phase qs[0], qs[1];

    gphase(π);
    inv @ gphase(π / 2);
    negctrl @ ctrl @ gphase(2 * π) qs[0], qs[1];

    #pragma {{"{string_to_bin("braket result amplitude '00', '01', '10', '11'")}";}}
    """
    simulator = OpenQASMStateVectorSimulator()
    result = simulator.run(Program(source=qasm))
    sv = [result.resultTypes[0].value[state] for state in ("00", "01", "10", "11")]
    assert np.allclose(sv, [-1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


@pytest.fixture
def sv_adder(pytester, stdgates):
    pytester.makefile(
        ".qasm",
        adder=f"""
            /*
             * quantum ripple-carry adder
             * Cuccaro et al, quant-ph/0410184
             * (adjusted for Braket)
             */
            OPENQASM 3;
            include "stdgates.inc";

            input uint[4] a_in;
            input uint[4] b_in;

            gate majority a, b, c {{
                cx c, b;
                cx c, a;
                ccx a, b, c;
            }}

            gate unmaj a, b, c {{
                ccx a, b, c;
                cx c, a;
                cx a, b;
            }}

            qubit cin;
            qubit[4] a;
            qubit[4] b;
            qubit cout;

            // set input states
            for i in [0: 3] {{
              if(bool(a_in[i])) x a[i];
              if(bool(b_in[i])) x b[i];
            }}

            // add a to b, storing result in b
            majority cin, b[3], a[3];
            for i in [3: -1: 1] {{ majority a[i], b[i - 1], a[i - 1]; }}
            cx a[0], cout;
            for i in [1: 3] {{ unmaj a[i], b[i - 1], a[i - 1]; }}
            unmaj cin, b[3], a[3];

            // todo: subtle bug when trying to get a result type for both at once
            #pragma {{"{string_to_bin("braket result probability cout, b")}";}}
            #pragma {{"{string_to_bin("braket result probability cout")}";}}
            #pragma {{"{string_to_bin("braket result probability b")}";}}
        """,
    )


@pytest.mark.xfail(reason="result types not translated for shots simulation until bdk")
def test_adder(sv_adder):
    simulator = OpenQASMStateVectorSimulator()
    inputs = {"a_in": 7, "b_in": 3}
    result = simulator.run(Program(source="adder.qasm", inputs=inputs), shots=100)
    expected_probs = np.zeros(2**5)
    expected_probs[10] = 1
    assert np.array_equal(result.resultTypes[0].value, expected_probs)


def test_adder_analytic(sv_adder):
    simulator = OpenQASMStateVectorSimulator()
    inputs = {"a_in": 7, "b_in": 3}
    result = simulator.run(Program(source="adder.qasm", inputs=inputs))
    expected_probs = np.zeros(2**5)
    expected_probs[10] = 1
    probs = np.outer(result.resultTypes[1].value, result.resultTypes[2].value).flatten()
    assert np.allclose(probs, expected_probs)


def test_no_reset():
    qasm = """
    qubit[2] qs;
    reset qs;
    """
    simulator = OpenQASMStateVectorSimulator()
    no_reset = "Quantum reset not implemented"
    with pytest.raises(NotImplementedError, match=no_reset):
        simulator.run(Program(source=qasm))


def test_no_measure():
    qasm = """
    qubit[2] qs;
    measure qs;
    """
    simulator = OpenQASMStateVectorSimulator()
    no_measure = "Measurements not implemented"
    with pytest.raises(NotImplementedError, match=no_measure):
        simulator.run(Program(source=qasm))
