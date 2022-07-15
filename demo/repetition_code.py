from braket.default_simulator import StateVectorSimulator
from braket.ir.openqasm import Program


ghz_qasm = """
OPENQASM 3;

qubit[3] q;
qubit[2] f;
output uint[3] cq;
output uint[2] cf;

// Inirtial state
// x q[0]; // uncomment to start with state 1

// Encoding 
cnot q[0], q[1];
cnot q[0], q[2];

// single Bit flip error
x q[1];

// Flag qubit 0
cnot q[0], f[0];
cnot q[2], f[0];

// Flag qubit 1
cnot q[1], f[1];
cnot q[2], f[1];

cf = measure f;

// Correct error based on error syndrome
if (cf==2){ 
  x q[0];
}
if (cf==1){ 
  x q[1];
}
if (cf==3){ 
  x q[2];
}

// measure q
cq = measure q;

"""


device = StateVectorSimulator()
program = Program(source=ghz_qasm)

result = device.run(program, shots=5, mcm=True)

for i in range(5):

  cf = result["cf"][i]
  print(f"error syndrome = {cf:02b}")
  cq = result["cq"][i]
  print(f"final sample   = {cq:03b}")

  print()


