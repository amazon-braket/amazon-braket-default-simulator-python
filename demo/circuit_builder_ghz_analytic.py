from braket.devices import LocalSimulator
from braket.ir.openqasm import Program

device = LocalSimulator("oq3_sv")

ghz_qasm = """
include "stdgates.inc";
qubit[3] q;

h q[0];
cx q[0], q[1];
cx q[1], q[2];

#pragma {"braket result probability";}
#pragma {"braket result amplitude '000', '111'";}
"""

ghz = Program(source=ghz_qasm)
result = device.run(ghz).result()
print(result)
print(result.result_types)
print(result.measurement_counts)
