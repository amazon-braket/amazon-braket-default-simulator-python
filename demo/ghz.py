from braket.ir.openqasm import Program
from braket.devices import LocalSimulator

device = LocalSimulator('braket_oq3_sv')

ghz_qasm = """
gate h a { U(π/2, 0, π) a; }
gate x a { U(π, 0, π) a; }
gate cx a, b { ctrl @ x a, b; }

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
