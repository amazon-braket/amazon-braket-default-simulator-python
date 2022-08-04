import pytest
from braket.ir.jaqcd import Probability

from braket.default_simulator.gate_operations import U
from braket.default_simulator.openqasm.circuit import Circuit


@pytest.mark.parametrize(
    "instructions, results, num_qubits",
    (
        (
            [U((0, 1, 2), 1, 1, 1, (0, 1))],
            [Probability()],
            3,
        ),
        (
            [U((0,), 1, 1, 1, ())],
            [],
            1,
        ),
    ),
)
def test_construct_circuit(instructions, results, num_qubits):
    circuit = Circuit(instructions, results)
    assert circuit.instructions == instructions
    assert circuit.results == results
    assert circuit.num_qubits == num_qubits
