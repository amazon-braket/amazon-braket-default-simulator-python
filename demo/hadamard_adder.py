import numpy as np
# from braket.devices import LocalSimulator
from braket.ir.openqasm import Program

adder_qasm = """
OPENQASM 3;
include "stdgates.inc";

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
qubit[4] b_copy;
qubit cout;
qubit[4] b;

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
"""


device = LocalSimulator("braket_oq3_sv")

adder = Program(source=adder_qasm)
num_shots = 10_000
result = device.run(adder, shots=num_shots).result()


def bit_string_to_int(bit_string):
    return bit_string @ (2 ** np.arange(bit_string.size)[::-1])


for shot in range(num_shots):
    measurement = result.measurements[shot]
    a = measurement[1:5]
    b = measurement[5:9]
    ans = measurement[9:]
    print(f"{bit_string_to_int(a)} + {bit_string_to_int(b)} = {bit_string_to_int(ans)}")
