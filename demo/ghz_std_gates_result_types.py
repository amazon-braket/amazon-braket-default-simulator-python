from braket.devices import LocalSimulator
from braket.ir.openqasm import Program

device = LocalSimulator("braket_oq3_sv")

ghz_qasm = """
include "stdgates.inc";
qubit p;
qubit[3] q;

h q[0];
cx q[0], q[1];
cx q[1], q[2];

//#pragma {"braket result state_vector";}
#pragma {"braket result probability";}
#pragma {"braket result probability q";}
#pragma {"braket result probability q[1]";}
#pragma {"braket result probability q[0:1]";}
//#pragma {"braket result amplitude '000', '111'";}
#pragma {"braket result expectation x(q[0]) @ z(q[2]) @ h(q[1])";}
#pragma {"braket result expectation x(q[2])";}
#pragma {"braket result variance x(q[0]) @ z(q[2]) @ h(q[1])";}
//#pragma {"braket result sample x(q[0]) @ z(q[2]) @ h(q[1])";}
#pragma {"braket result expectation hermitian([[0, -1im], [1im, 0]]) q[0]";}
"""

ghz = Program(source=ghz_qasm)
result = device.run(ghz, shots=10).result()
# print(result.result_types)
# print(result.result_types[0])
for rt in result.result_types:
    print(f"{rt.type}: {rt.value}")
# print(np.round(result.result_types[0].value, 4))
