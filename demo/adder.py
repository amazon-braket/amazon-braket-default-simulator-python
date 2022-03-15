import numpy as np

from braket.ir.openqasm import Program
from braket.devices import LocalSimulator

device = LocalSimulator('braket_oq3_sv')


def adder_qasm(a: int, b: int):
    return f"""
/*
 * quantum ripple-carry adder
 * Cuccaro et al, quant-ph/0410184
 * (adjusted for Braket)
 */
OPENQASM 3;
include "stdgates.inc";

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
bit[5] ans;
uint[4] a_in = {a};
uint[4] b_in = {b};

// initialize qubits
reset cin;
reset a;
reset b;
reset cout;

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

// measure results
ans[0] = measure cout[0];
ans[1:4] = measure b[0:3];
"""


for _ in range(10):
    a, b = np.random.randint(0, 16, 2)
    adder = Program(source=adder_qasm(a, b))
    result = device.run(adder, shots=1).result()
    ans = result.bit_variables["ans"][0]
    print(f"{a} + {b} = {int(ans, base=2)}")
