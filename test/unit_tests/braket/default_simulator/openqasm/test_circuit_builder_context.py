import pytest
from openqasm3.ast import Identifier

from braket.default_simulator.openqasm.circuit_builder_context import CircuitBuilderContext


def test_cannot_do_quantum_ops():
    context = CircuitBuilderContext()

    with pytest.raises(NotImplementedError):
        context.reset_qubits(Identifier("q"))

    with pytest.raises(NotImplementedError):
        context.measure_qubits(Identifier("q"))
