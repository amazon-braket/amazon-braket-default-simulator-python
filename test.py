# from braket.default_simulator.branched_simulator import BranchedSimulator
from collections import Counter, defaultdict
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.devices import LocalSimulator

qasm_source = """
OPENQASM 3.0;
qubit[11] __qubits__;
h __qubits__[10];

for int i in [0:9]{
    cnot __qubits__[10], __qubits__[i];
}
"""

program = OpenQASMProgram(source=qasm_source, inputs={})

python_result = LocalSimulator("braket_sv").run(program, shots=1000).result()

print(python_result.measurement_counts)