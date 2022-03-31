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


@pytest.fixture
def hadamard_adder(pytester, stdgates):
    pytester.makefile(
        ".qasm",
        hadamard_adder="""
            /*
             * quantum ripple-carry adder
             * Cuccaro et al, quant-ph/0410184
             * (adjusted for Braket)
             */
            OPENQASM 3;
            include "stdgates.inc";

            output uint[4] a_in;
            output uint[4] b_in;
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
            qubit[4] b_copy;
            qubit cout;

            // initialize qubits
            reset cin;
            reset a;
            reset b;
            reset b_copy;
            reset cout;

            // set input states
            for i in [0: 3] {
              h a[i];
              h b[i];
              cx b[i], b_copy[i];
            }

            // add a to b, storing result in b
            majority cin, b[3], a[3];
            for i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
            cx a[0], cout;
            for i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
            unmaj cin, b[3], a[3];

            // measure results
            a_in = measure a;
            b_in = measure b_copy;
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


def test_input_output_types():
    device = LocalSimulator("braket_oq3_sv")
    result = device.run(
        Program(
            source="""
                input bit[4] bits_in;
                input bit bit_in;
                input int[8] int_in;
                input float[16] float_in;
                input array[uint[8], 4] array_in;

                output bit[4] bits_out;
                output bit bit_out;
                output int[8] int_out;
                output float[16] float_out;
                output array[uint[8], 4] array_out;

                bits_out = bits_in;
                bit_out = bit_in;
                int_out = int_in;
                float_out = float_in;
                array_out = array_in;
            """,
            inputs={
                "bits_in": "1010",
                "bit_in": True,
                "int_in": 12,
                "float_in": np.pi,
                "array_in": [1, 2, 3, 4],
            },
        ),
        shots=2,
    ).result()
    print(result.output_variables)
    assert set(result.output_variables.keys()) == {
        "bits_out",
        "bit_out",
        "int_out",
        "float_out",
        "array_out",
    }
    assert np.array_equal(
        result.output_variables["bits_out"],
        np.full(2, "1010"),
    )
    assert np.array_equal(
        result.output_variables["bit_out"],
        np.full(2, True),
    )
    assert np.array_equal(
        result.output_variables["int_out"],
        np.full(2, 12),
    )
    assert np.array_equal(
        result.output_variables["float_out"],
        np.full(2, np.pi, dtype=np.float16),
    )
    assert np.array_equal(
        result.output_variables["array_out"],
        [[1, 2, 3, 4]] * 2,
    )


# def test_shots_equals_zero(hadamard_adder):
#     device = LocalSimulator("braket_oq3_sv")
#     # result = device.run(
#     #     Program(source="hadamard_adder.qasm"),
#     #     shots=0,
#     # )
#     # print(result)