# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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


def test_add_measure_rejects_duplicate_qubit_by_default():
    circuit = Circuit()
    circuit.add_measure((0,), [0])
    with pytest.raises(ValueError, match="Qubit 0 is already measured or captured."):
        circuit.add_measure((0,), [1])


def test_add_measure_returns_explicit_classical_indices():
    circuit = Circuit()
    assert circuit.add_measure((0, 1), [3, 5]) == [3, 5]
    assert circuit.target_classical_indices == [3, 5]


def test_add_measure_returns_auto_assigned_classical_indices():
    circuit = Circuit()
    assert circuit.add_measure((0, 1)) == [0, 1]
    assert circuit.add_measure((2,)) == [2]
    assert circuit.target_classical_indices == [0, 1, 2]


def test_add_measure_returns_indices_on_remeasure():
    circuit = Circuit()
    assert circuit.add_measure((0,), [0]) == [0]
    assert circuit.add_measure((1,), [0], allow_remeasure=True) == [0]
    assert circuit.measured_qubits == [1]
    assert circuit.target_classical_indices == [0]


def test_add_measure_rejects_duplicate_classical_index_by_default():
    """Without remeasure, reusing a classical index would silently produce a
    duplicated column, so it is rejected."""
    circuit = Circuit()
    circuit.add_measure((0,), [0])
    with pytest.raises(ValueError, match="Classical bit 0 is already assigned"):
        circuit.add_measure((1,), [0])
