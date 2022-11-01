from braket.default_simulator.openqasm.interpreter import Interpreter

qasm = """
OPENQASM 3;

input float theta;
input float phi;

qubit[2] q;
bit[2] c;

h q[0];
cnot q[0], q[1];
rx(theta) q[0];
ry(phi) q;

#pragma braket result state_vector
#pragma braket result state_vector all
#pragma braket result state_vector q[0]
"""
interpreter = Interpreter()
circuit = interpreter.build_circuit(qasm)

for instruction in circuit.instructions:
    print(
        f"{type(instruction).__name__}({getattr(instruction, '_angle', None)}) "
        f"{', '.join(map(str, instruction._targets))}"
    )
