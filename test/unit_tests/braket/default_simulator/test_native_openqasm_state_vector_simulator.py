# import numpy as np
# import pytest
# from braket.ir.jaqcd import (
#     Amplitude,
#     DensityMatrix,
#     Expectation,
#     Probability,
#     StateVector,
#     Variance,
# )
# from braket.ir.openqasm import Program
#
# from braket.default_simulator.openqasm_native_state_vector_simulator import (
#     OpenQASMNativeStateVectorSimulator,
# )
#
#
# def string_to_bin(string):
#     """workaround for unsupported pragmas"""
#     return "".join(np.binary_repr(ord(x), 8) for x in string)
#
#
# @pytest.fixture
# def ghz(pytester, stdgates):
#     pytester.makefile(
#         ".qasm",
#         ghz="""
#             include "stdgates.inc";
#
#             qubit[3] q;
#             bit[3] c;
#
#             h q[0];
#             cx q[0], q[1];
#             cx q[1], q[2];
#
#             c = measure q;
#         """,
#     )
#
#
# @pytest.fixture
# def adder(pytester, stdgates):
#     pytester.makefile(
#         ".qasm",
#         adder="""
#             /*
#              * quantum ripple-carry adder
#              * Cuccaro et al, quant-ph/0410184
#              * (adjusted for Braket)
#              */
#             OPENQASM 3;
#             include "stdgates.inc";
#
#             input uint[4] a_in;
#             input uint[4] b_in;
#             output bit[5] ans;
#
#             gate majority a, b, c {
#                 cx c, b;
#                 cx c, a;
#                 ccx a, b, c;
#             }
#
#             gate unmaj a, b, c {
#                 ccx a, b, c;
#                 cx c, a;
#                 cx a, b;
#             }
#
#             qubit cin;
#             qubit[4] a;
#             qubit[4] b;
#             qubit cout;
#
#             // initialize qubits
#             reset cin;
#             reset a;
#             reset b;
#             reset cout;
#
#             // set input states
#             for i in [0: 3] {
#               if(bool(a_in[i])) x a[i];
#               if(bool(b_in[i])) x b[i];
#             }
#
#             // add a to b, storing result in b
#             majority cin, b[3], a[3];
#             for i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
#             cx a[0], cout;
#             for i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
#             unmaj cin, b[3], a[3];
#
#             // measure results
#             ans[0] = measure cout;
#             ans[1:4] = measure b[0:3];
#         """,
#     )
#
#
# @pytest.fixture
# def hadamard_adder(pytester, stdgates):
#     pytester.makefile(
#         ".qasm",
#         hadamard_adder="""
#             /*
#              * quantum ripple-carry adder
#              * Cuccaro et al, quant-ph/0410184
#              * (adjusted for Braket)
#              */
#             OPENQASM 3;
#             include "stdgates.inc";
#
#             output uint[4] a_in;
#             output uint[4] b_in;
#             output bit[5] ans;
#
#             gate majority a, b, c {
#                 cx c, b;
#                 cx c, a;
#                 ccx a, b, c;
#             }
#
#             gate unmaj a, b, c {
#                 ccx a, b, c;
#                 cx c, a;
#                 cx a, b;
#             }
#
#             qubit cin;
#             qubit[4] a;
#             qubit[4] b;
#             qubit[4] b_copy;
#             qubit cout;
#
#             // initialize qubits
#             reset cin;
#             reset a;
#             reset b;
#             reset b_copy;
#             reset cout;
#
#             // set input states
#             for i in [0: 3] {
#               h a[i];
#               h b[i];
#               cx b[i], b_copy[i];
#             }
#
#             // add a to b, storing result in b
#             majority cin, b[3], a[3];
#             for i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
#             cx a[0], cout;
#             for i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
#             unmaj cin, b[3], a[3];
#
#             // measure results
#             a_in = measure a;
#             b_in = measure b_copy;
#             ans[0] = measure cout;
#             ans[1:4] = measure b[0:3];
#         """,
#     )
#
#
# @pytest.mark.parametrize("shots", (1, 2, 10))
# def test_ghz(ghz, shots):
#     simulator = OpenQASMNativeStateVectorSimulator()
#     program = Program(source="ghz.qasm")
#     result = simulator.run(program, shots=shots)
#
#     assert len(result.outputVariables["c"]) == shots
#     for outcome in result.outputVariables["c"]:
#         assert outcome in ("000", "111")
#
#
# def test_adder(adder):
#     simulator = OpenQASMNativeStateVectorSimulator()
#     for _ in range(3):
#         a, b = np.random.randint(0, 16, 2)
#         inputs = {"a_in": a, "b_in": b}
#         program = Program(source="adder.qasm", inputs=inputs)
#         result = simulator.run(program, shots=1)
#         ans = int(result.outputVariables["ans"][0], base=2)
#         assert a + b == ans
#
#
# def test_input_output_types():
#     simulator = OpenQASMNativeStateVectorSimulator()
#     result = simulator.run(
#         Program(
#             source="""
#                 input bit[4] bits_in;
#                 input bit bit_in;
#                 input int[8] int_in;
#                 input float[16] float_in;
#                 input array[uint[8], 4] array_in;
#
#                 output bit[4] bits_out;
#                 output bit bit_out;
#                 output int[8] int_out;
#                 output float[16] float_out;
#                 output array[uint[8], 4] array_out;
#
#                 bits_out = bits_in;
#                 bit_out = bit_in;
#                 int_out = int_in;
#                 float_out = float_in;
#                 array_out = array_in;
#             """,
#             inputs={
#                 "bits_in": "1010",
#                 "bit_in": True,
#                 "int_in": 12,
#                 "float_in": np.pi,
#                 "array_in": [1, 2, 3, 4],
#             },
#         ),
#         shots=2,
#     )
#     assert set(result.outputVariables.keys()) == {
#         "bits_out",
#         "bit_out",
#         "int_out",
#         "float_out",
#         "array_out",
#     }
#     assert np.array_equal(
#         result.outputVariables["bits_out"],
#         np.full(2, "1010"),
#     )
#     assert np.array_equal(
#         result.outputVariables["bit_out"],
#         np.full(2, True),
#     )
#     assert np.array_equal(
#         result.outputVariables["int_out"],
#         np.full(2, 12),
#     )
#     assert np.array_equal(
#         result.outputVariables["float_out"],
#         np.full(2, np.pi, dtype=np.float16),
#     )
#     assert np.array_equal(
#         result.outputVariables["array_out"],
#         [[1, 2, 3, 4]] * 2,
#     )
#
#
# def test_result_types_analytic(stdgates):
#     simulator = OpenQASMNativeStateVectorSimulator()
#     qasm = f"""
#     include "stdgates.inc";
#
#     qubit[3] q;
#     bit[3] c;
#
#     h q[0];
#     cx q[0], q[1];
#     cx q[1], q[2];
#     x q[2];
#
#     // {{ 001: .5, 110: .5 }}
#
#     #pragma {{"{string_to_bin("braket result state_vector")}";}}
#     #pragma {{"{string_to_bin("braket result probability")}";}}
#     #pragma {{"{string_to_bin("braket result probability q")}";}}
#     #pragma {{"{string_to_bin("braket result probability q[0]")}";}}
#     #pragma {{"{string_to_bin("braket result probability q[0:1]")}";}}
#     #pragma {{"{string_to_bin("braket result probability q[{0, 2, 1}]")}";}}
#     #pragma {{"{string_to_bin('braket result amplitude "001", "110"')}";}}
#     #pragma {{"{string_to_bin("braket result density_matrix")}";}}
#     #pragma {{"{string_to_bin("braket result density_matrix q")}";}}
#     #pragma {{"{string_to_bin("braket result density_matrix q[0]")}";}}
#     #pragma {{"{string_to_bin("braket result density_matrix q[0:1]")}";}}
#     #pragma {{"{string_to_bin("braket result density_matrix q[{0, 2, 1}]")}";}}
#     #pragma {{"{string_to_bin("braket result expectation z(q[0])")}";}}
#     #pragma {{"{string_to_bin("braket result variance x(q[0]) @ z(q[2]) @ h(q[1])")}";}}
#     #pragma {{"{string_to_bin(
#         "braket result expectation hermitian([[0, -1im], [1im, 0]]) q[0]"
#     )}";}}
#     """
#     program = Program(source=qasm)
#     result = simulator.run(program, shots=0)
#     for rt in result.resultTypes:
#         print(f"{rt.type}: {rt.value}")
#
#     print(np.abs(np.round(result.resultTypes[11].value, 2).astype(float)))
#
#     result_types = result.resultTypes
#
#     assert result_types[0].type == StateVector()
#     assert result_types[1].type == Probability()
#     assert result_types[2].type == Probability(targets=(0, 1, 2))
#     assert result_types[3].type == Probability(targets=(0,))
#     assert result_types[4].type == Probability(targets=(0, 1))
#     assert result_types[5].type == Probability(targets=(0, 2, 1))
#     assert result_types[6].type == Amplitude(states=("001", "110"))
#     assert result_types[7].type == DensityMatrix()
#     assert result_types[8].type == DensityMatrix(targets=(0, 1, 2))
#     assert result_types[9].type == DensityMatrix(targets=(0,))
#     assert result_types[10].type == DensityMatrix(targets=(0, 1))
#     assert result_types[11].type == DensityMatrix(targets=(0, 2, 1))
#     assert result_types[12].type == Expectation(observable=("z",), targets=(0,))
#     assert result_types[13].type == Variance(observable=("x", "z", "h"), targets=(0, 2, 1))
#     assert result_types[14].type == Expectation(
#         observable=([[[0, 0], [0, -1]], [[0, 1], [0, 0]]],),
#         targets=(0,),
#     )
#
#     assert np.allclose(
#         result_types[0].value,
#         [0, 1 / np.sqrt(2), 0, 0, 0, 0, 1 / np.sqrt(2), 0],
#     )
#     assert np.allclose(
#         result_types[1].value,
#         [0, 0.5, 0, 0, 0, 0, 0.5, 0],
#     )
#     assert np.allclose(
#         result_types[2].value,
#         [0, 0.5, 0, 0, 0, 0, 0.5, 0],
#     )
#     assert np.allclose(
#         result_types[3].value,
#         [0.5, 0.5],
#     )
#     assert np.allclose(
#         result_types[4].value,
#         [0.5, 0, 0, 0.5],
#     )
#     assert np.allclose(
#         result_types[5].value,
#         [0, 0, 0.5, 0, 0, 0.5, 0, 0],
#     )
#     assert np.isclose(result_types[6].value["001"], 1 / np.sqrt(2))
#     assert np.isclose(result_types[6].value["110"], 1 / np.sqrt(2))
#     assert np.allclose(
#         result_types[7].value,
#         [
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0.5, 0, 0, 0, 0, 0.5, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0.5, 0, 0, 0, 0, 0.5, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#     )
#     assert np.allclose(
#         result_types[8].value,
#         [
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0.5, 0, 0, 0, 0, 0.5, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0.5, 0, 0, 0, 0, 0.5, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#     )
#     assert np.allclose(
#         result_types[9].value,
#         np.eye(2) * 0.5,
#     )
#     assert np.allclose(
#         result_types[10].value,
#         [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]],
#     )
#     assert np.allclose(
#         result_types[11].value,
#         [
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0.5, 0, 0, 0.5, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0.5, 0, 0, 0.5, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#     )
#     assert np.allclose(result_types[12].value, 0)
#     assert np.allclose(result_types[13].value, 1)
#     assert np.allclose(result_types[14].value, 0)
#
#
# def test_invalid_stanard_observable_target():
#     qasm = f"""
#     qubit[2] qs;
#     #pragma {{"{string_to_bin("braket result variance x(qs)")}";}}
#     """
#     simulator = OpenQASMNativeStateVectorSimulator()
#     program = Program(source=qasm)
#
#     must_be_one_qubit = "Standard observable target must be exactly 1 qubit."
#
#     with pytest.raises(ValueError, match=must_be_one_qubit):
#         simulator.run(program, shots=0)
#
#
# # def test_shots_equals_zero(hadamard_adder):
# #     device = LocalSimulator("braket_oq3_sv")
# #     # result = device.run(
# #     #     Program(source="hadamard_adder.qasm"),
# #     #     shots=0,
# #     # )
# #     # print(result)
#
#
# # def test_results_not_supported():
# #     qasm = """
# #     include "stdgates.inc";
# #     qubit[3] q;
# #
# #     h q[0];
# #     cx q[0], q[1];
# #     cx q[1], q[2];
# #
# #     #pragma {"braket result state_vector";}
# #     """
# #     not_supported = "result type FakeStateVector is not
# #     supported by OpenQASMNativeStateVectorSimulator"
# #     with pytest.raises(TypeError, match=not_supported)
