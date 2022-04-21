from braket.devices import LocalSimulator
from braket.ir.openqasm import Program

device = LocalSimulator("braket_oq3_native_sv")

ghz_qasm = """
include "stdgates.inc";
output bit[5] c;

def entangle(qubit q1, qubit q2) {
    cx q1, q2;
}

const int[5] n = 5;
qubit[n] qs;

h qs[0];
for i in [1:n-1] {
    entangle(qs[i-1], qs[i]);
}

c = measure qs;
"""

ghz = Program(source=ghz_qasm)
result = device.run(ghz, shots=100).result()
print(result.output_variables)
