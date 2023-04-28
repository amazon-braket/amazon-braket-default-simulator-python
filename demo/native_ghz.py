from braket.ir.openqasm import Program

from braket.default_simulator.openqasm_native_state_vector_matrix_simulator import OpenQASMNativeStateVectorSimulator

ghz_qasm = """
gate h a { U(π/2, 0, π) a; }
gate x a { U(π, 0, π) a; }
gate cx a, b { ctrl @ x a, b; }

qubit[3] q;
output uint first_two;
output uint last_one;

output float x;
x = 2;

h q[0];
cx q[0], q[1];

first_two = measure q[0:1];

if (first_two) {
  x q[-1];
}
last_one = measure q[2];
"""


device = OpenQASMNativeStateVectorSimulator()
program = Program(source=ghz_qasm)
num_shots = 100

result = device.run(program, shots=num_shots)
print(result)
# exit()

for i in range(num_shots):
    first_two = result["first_two"][i]
    last_one = result["last_one"][i]

    print(f"measured int for q[{{0, 1}}]: {first_two}, q[2]: {last_one}")
