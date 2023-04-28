from braket.default_simulator.openqasm_native_state_vector_matrix_simulator import OpenQASMNativeStateVectorSimulator
from braket.ir.openqasm import Program as BraketProgram
qasm = '''
OPENQASM 3.0;
output bit f;
def foo() -> bit {
    h q[0];
    bit i = 0;
    i = measure q[0];
    return i;
}
qubit[10] q;
f = foo();
'''
result = OpenQASMNativeStateVectorSimulator().run(BraketProgram(source=qasm), shots=5)
print(result)
