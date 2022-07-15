from braket.ir.openqasm import Program

from braket.default_simulator.openqasm_native_state_vector_matrix_simulator import OpenQASMNativeStateVectorSimulator

ghz_qasm = """
OPENQASM 3;

qubit[3] q;
qubit[2] f;
bit[3] cq;
bit[2] cf;

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

uint cf = measure f;

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
uint cq = measure q;

"""


device = OpenQASMNativeStateVectorSimulator()
program = Program(source=ghz_qasm)

for _ in range(5):
  result = device.run(program, shots=1)
  cf = result.get_value("cf").value
  print(f"error syndrome = {cf:02b}")
  cq = result.get_value("cq").value
  print(f"final sample   = {cq:03b}")

  print()


