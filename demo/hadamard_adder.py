from braket.devices import LocalSimulator
from braket.ir.openqasm import Program

adder_qasm = """
/*
 * quantum ripple-carry adder
 * Cuccaro et al, quant-ph/0410184
 * (adjusted for Braket)
 */
OPENQASM 3;
include "stdgates.inc";

output bit[4] a_in;
output bit[4] b_in;
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
"""


device = LocalSimulator("braket_oq3_sv")

adder = Program(source=adder_qasm)
num_shots = 10
result = device.run(adder, shots=num_shots).result()

for shot in range(num_shots):
    a = result.output_variables["a_in"][shot]
    b = result.output_variables["b_in"][shot]
    ans = result.output_variables["ans"][shot]
    print(f"{int(a, base=2)} + {int(b, base=2)} = {int(ans, base=2)}")
