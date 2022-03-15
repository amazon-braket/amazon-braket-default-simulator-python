from braket.ir.openqasm import Program
from braket.devices import LocalSimulator

device = LocalSimulator('braket_oq3_sv')

ghz_qasm = """
include "stdgates.inc";

qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];
cx q[1], q[2];

c = measure q;
"""


ghz = Program(source=ghz_qasm)
result = device.run(ghz, shots=10).result()

print(result.bit_variables)
