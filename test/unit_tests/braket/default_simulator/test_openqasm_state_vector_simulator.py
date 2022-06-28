import numpy as np
import pytest
from braket.ir.jaqcd import (
    Amplitude,
    DensityMatrix,
    Expectation,
    Probability,
    StateVector,
    Variance,
)
from braket.ir.openqasm import Program

from braket.default_simulator.openqasm_state_vector_simulator import OpenQASMStateVectorSimulator


def test_gphase():
    qasm = """
    qubit[2] qs;

    int[8] two = 2;

    gate x a { U(π, 0, π) a; }
    gate cx c, a { ctrl @ x c, a; }
    gate phase c, a {
        gphase(π/2);
        pow(1) @ ctrl(two) @ gphase(π) c, a;
    }
    gate h a { U(π/2, 0, π) a; }

    inv @ U(π/2, 0, π) qs[0];
    cx qs[0], qs[1];
    phase qs[0], qs[1];

    gphase(π);
    inv @ gphase(π / 2);
    negctrl @ ctrl @ gphase(2 * π) qs[0], qs[1];

    #pragma braket result amplitude '00', '01', '10', '11'
    """
    simulator = OpenQASMStateVectorSimulator()
    result = simulator.run(Program(source=qasm))
    sv = [result.resultTypes[0].value[state] for state in ("00", "01", "10", "11")]
    assert np.allclose(sv, [-1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


@pytest.fixture
def sv_adder(pytester, stdgates):
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

            // set input states
            for int[8] i in [0: 3] {
              if(bool(a_in[i])) x a[i];
              if(bool(b_in[i])) x b[i];
            }

            // add a to b, storing result in b
            majority cin, b[3], a[3];
            for int[8] i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
            cx a[0], cout;
            for int[8] i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
            unmaj cin, b[3], a[3];

            // todo: subtle bug when trying to get a result type for both at once
            #pragma braket result probability cout, b
            #pragma braket result probability cout
            #pragma braket result probability b
        """,
    )


@pytest.mark.xfail(reason="result types not translated for shots simulation until bdk")
def test_adder(sv_adder):
    simulator = OpenQASMStateVectorSimulator()
    inputs = {"a_in": 7, "b_in": 3}
    result = simulator.run(Program(source="adder.qasm", inputs=inputs), shots=100)
    expected_probs = np.zeros(2**5)
    expected_probs[10] = 1
    assert np.allclose(result.resultTypes[0].value, expected_probs)


def test_adder_analytic(sv_adder):
    simulator = OpenQASMStateVectorSimulator()
    inputs = {"a_in": 7, "b_in": 3}
    result = simulator.run(Program(source="adder.qasm", inputs=inputs))
    expected_probs = np.zeros(2**5)
    expected_probs[10] = 1
    probs = np.outer(result.resultTypes[1].value, result.resultTypes[2].value).flatten()
    assert np.allclose(probs, expected_probs)


def test_result_types_analytic(stdgates):
    simulator = OpenQASMStateVectorSimulator()
    qasm = """
    include "stdgates.inc";

    qubit[3] q;
    bit[3] c;

    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    x q[2];

    // {{ 001: .5, 110: .5 }}

    #pragma braket result state_vector
    #pragma braket result probability
    #pragma braket result probability q
    #pragma braket result probability q[0]
    #pragma braket result probability q[0:1]
    #pragma braket result probability q[{0, 2, 1}]
    #pragma braket result amplitude "001", "110"
    #pragma braket result density_matrix
    #pragma braket result density_matrix q
    #pragma braket result density_matrix q[0]
    #pragma braket result density_matrix q[0:1]
    #pragma braket result density_matrix q[0], q[1]
    #pragma braket result density_matrix q[{0, 2, 1}]
    #pragma braket result expectation z(q[0])
    #pragma braket result variance x(q[0]) @ z(q[2]) @ h(q[1])
    #pragma braket result expectation hermitian([[0, -1im], [0 + 1im, 0]]) q[0]
    """
    program = Program(source=qasm)
    result = simulator.run(program, shots=0)
    for rt in result.resultTypes:
        print(f"{rt.type}: {rt.value}")

    result_types = result.resultTypes

    assert result_types[0].type == StateVector()
    assert result_types[1].type == Probability()
    assert result_types[2].type == Probability(targets=(0, 1, 2))
    assert result_types[3].type == Probability(targets=(0,))
    assert result_types[4].type == Probability(targets=(0, 1))
    assert result_types[5].type == Probability(targets=(0, 2, 1))
    assert result_types[6].type == Amplitude(states=("001", "110"))
    assert result_types[7].type == DensityMatrix()
    assert result_types[8].type == DensityMatrix(targets=(0, 1, 2))
    assert result_types[9].type == DensityMatrix(targets=(0,))
    assert result_types[10].type == DensityMatrix(targets=(0, 1))
    assert result_types[11].type == DensityMatrix(targets=(0, 1))
    assert result_types[12].type == DensityMatrix(targets=(0, 2, 1))
    assert result_types[13].type == Expectation(observable=("z",), targets=(0,))
    assert result_types[14].type == Variance(observable=("x", "z", "h"), targets=(0, 2, 1))
    assert result_types[15].type == Expectation(
        observable=([[[0, 0], [0, -1]], [[0, 1], [0, 0]]],),
        targets=(0,),
    )

    assert np.allclose(
        result_types[0].value,
        [0, 1 / np.sqrt(2), 0, 0, 0, 0, 1 / np.sqrt(2), 0],
    )
    assert np.allclose(
        result_types[1].value,
        [0, 0.5, 0, 0, 0, 0, 0.5, 0],
    )
    assert np.allclose(
        result_types[2].value,
        [0, 0.5, 0, 0, 0, 0, 0.5, 0],
    )
    assert np.allclose(
        result_types[3].value,
        [0.5, 0.5],
    )
    assert np.allclose(
        result_types[4].value,
        [0.5, 0, 0, 0.5],
    )
    assert np.allclose(
        result_types[5].value,
        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
    )
    assert np.isclose(result_types[6].value["001"], 1 / np.sqrt(2))
    assert np.isclose(result_types[6].value["110"], 1 / np.sqrt(2))
    assert np.allclose(
        result_types[7].value,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert np.allclose(
        result_types[8].value,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert np.allclose(
        result_types[9].value,
        np.eye(2) * 0.5,
    )
    assert np.allclose(
        result_types[10].value,
        [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]],
    )
    assert np.allclose(
        result_types[11].value,
        [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]],
    )
    assert np.allclose(
        result_types[12].value,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert np.allclose(result_types[13].value, 0)
    assert np.allclose(result_types[14].value, 1)
    assert np.allclose(result_types[15].value, 0)


def test_invalid_standard_observable_target():
    qasm = """
    qubit[2] qs;
    #pragma braket result variance x(qs)
    """
    simulator = OpenQASMStateVectorSimulator()
    program = Program(source=qasm)

    must_be_one_qubit = "Standard observable target must be exactly 1 qubit."

    with pytest.raises(ValueError, match=must_be_one_qubit):
        simulator.run(program, shots=0)
