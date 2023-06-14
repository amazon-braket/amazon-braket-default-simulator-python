from braket.ir.openqasm import Program as BraketProgram

from braket.default_simulator import StateVectorSimulator

qasm = """
OPENQASM 3.0;
output bit[3] f;
def foo() -> bit[3] {
    h q[0];
    cnot q[0], q[1];
    cnot q[1], q[2];
    reset q[1];
    bit[3] i;
    i = measure q;
    return i;
}
qubit[3] q;
f = foo();
"""
result = StateVectorSimulator().run(BraketProgram(source=qasm), shots=5, mcm=True)
print(result)
