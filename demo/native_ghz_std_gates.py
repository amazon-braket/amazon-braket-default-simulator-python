from braket.devices import LocalSimulator
from braket.ir.openqasm import Program

device = LocalSimulator("braket_oq3_native_sv")

ghz_qasm = """
include "stdgates.inc";
output bit[3] c;

qubit[3] q;

h q[0];
cx q[0], q[1];
cx q[1], q[2];

c = measure q;
"""

ghz = Program(source=ghz_qasm)
result = device.run(ghz, shots=100).result()
print(result.output_variables)
