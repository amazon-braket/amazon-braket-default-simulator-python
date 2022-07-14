from braket.ir.openqasm import Program

from braket.default_simulator.openqasm_native_state_vector_matrix_simulator import OpenQASMNativeStateVectorSimulator

ghz_qasm = """
gate h a { U(π/2, 0, π) a; }
gate x a { U(π, 0, π) a; }
gate cx a, b { ctrl @ x a, b; }

qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];

uint first_two = measure q[0:1];

if (first_two) {
  x q[-1];
}
uint last_one = measure q[2];
"""


device = OpenQASMNativeStateVectorSimulator()
program = Program(source=ghz_qasm)

for _ in range(100):
    result = device.run(program)
    first_two = result.get_value("first_two").value
    last_one = result.get_value("last_one").value

    print(f"measured int for q[{{0, 1}}]: {first_two}, q[2]: {last_one}")
