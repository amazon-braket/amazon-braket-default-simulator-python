from braket.default_simulator.branched_simulator import BranchedSimulator
from collections import Counter, defaultdict
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.devices import LocalSimulator

qasm_source = """
OPENQASM 3.0;
qubit[9] __qubits__;
h __qubits__[2];

cnot __qubits__[2], __qubits__[1];

"""

program = OpenQASMProgram(source=qasm_source, inputs={})

python_result = LocalSimulator("braket_sv_branched_python").run(program, shots=1000).result()

print(python_result.measurement_counts)