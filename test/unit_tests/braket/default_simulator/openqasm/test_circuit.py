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
from braket.default_simulator.openqasm.circuit import Circuit, ClassicalRegister


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


def test_declare_register():
    circuit = Circuit()
    c = circuit.declare_register("c", 3)
    d = circuit.declare_register("d", 1)
    assert circuit.classical_registers == [c, d]
    assert (c.name, c.size, c.order, c.sources) == ("c", 3, 0, [None, None, None])
    assert (d.name, d.size, d.order, d.sources) == ("d", 1, 1, [None])
    assert circuit.measurement_slots == []
    assert circuit.measured_qubits == []


def test_registers_with_same_name_are_distinct():
    circuit = Circuit()
    outer = circuit.declare_register("b", 1)
    inner = circuit.declare_register("b", 1)
    assert outer is not inner
    assert outer != inner
    assert len({outer, inner}) == 2


def test_add_measure_into_register_elements():
    circuit = Circuit()
    c = circuit.declare_register("c", 3)
    bound = circuit.add_measure((5, 7), c, [2, 0])
    assert bound == [(c, 2), (c, 0)]
    assert c.sources == [7, None, 5]
    assert circuit.qubit_set == {5, 7}
    assert circuit.measurement_slots == [(c, 0, 7), (c, 2, 5)]
    assert circuit.measured_qubits == [7, 5]


def test_add_measure_whole_register_defaults_to_leading_elements():
    circuit = Circuit()
    c = circuit.declare_register("c", 3)
    assert circuit.add_measure((1, 2), c) == [(c, 0), (c, 1)]
    assert c.sources == [1, 2, None]


def test_remeasure_replaces_source():
    circuit = Circuit()
    b = circuit.declare_register("b", 1)
    circuit.add_measure((0,), b, [0])
    circuit.add_measure((1,), b, [0])
    assert b.sources == [1]
    assert circuit.measured_qubits == [1]


def test_qubit_may_source_several_elements():
    circuit = Circuit()
    b = circuit.declare_register("b", 2)
    circuit.add_measure((0,), b, [0])
    circuit.add_measure((0,), b, [1])
    assert circuit.measured_qubits == [0, 0]


def test_bind_out_of_range_raises():
    circuit = Circuit()
    b = circuit.declare_register("b", 1)
    with pytest.raises(IndexError, match="index 2 out of range for register of length 1 `b`"):
        circuit.add_measure((0,), b, [2])


def test_anonymous_register_grows_on_demand():
    circuit = Circuit()
    assert circuit.classical_registers == []
    assert circuit.add_measure((1, 0)) == [
        (circuit.anonymous_register(), 0),
        (circuit.anonymous_register(), 1),
    ]
    circuit.add_measure((2,))
    anonymous = circuit.anonymous_register()
    assert circuit.classical_registers == [anonymous]
    assert anonymous.name is None
    assert anonymous.sources == [1, 0, 2]
    assert circuit.measured_qubits == [1, 0, 2]


def test_add_measure_by_index_rejects_duplicate_qubit_by_default():
    circuit = Circuit()
    circuit.add_measure_by_index((0,), [0])
    with pytest.raises(ValueError, match="Qubit 0 is already measured or captured."):
        circuit.add_measure_by_index((0,), [1])


def test_add_measure_by_index_binds_elements_of_anonymous_register():
    """Explicit indices address elements of the anonymous register; the derived
    columns come out in index order."""
    circuit = Circuit()
    circuit.add_measure_by_index((5, 7), [2, 0], allow_remeasure=True)
    (anonymous,) = circuit.classical_registers
    assert anonymous is circuit.anonymous_register()
    assert anonymous.sources == [7, None, 5]
    assert circuit.measured_qubits == [7, 5]
    assert circuit.target_classical_indices == [0, 2]
    assert circuit.qubit_set == {5, 7}


def test_add_measure_by_index_appends_after_bound_elements():
    circuit = Circuit()
    circuit.add_measure_by_index((3,), [0], allow_remeasure=True)
    circuit.add_measure_by_index((1, 2), allow_remeasure=True)
    assert circuit.anonymous_register().sources == [3, 1, 2]
    assert circuit.target_classical_indices == [0, 1, 2]


def test_add_measure_by_index_remeasure_replaces_source():
    circuit = Circuit()
    circuit.add_measure_by_index((0,), [1], allow_remeasure=True)
    circuit.add_measure_by_index((4,), [1], allow_remeasure=True)
    assert circuit.anonymous_register().sources == [None, 4]
    assert circuit.measured_qubits == [4]
    assert circuit.target_classical_indices == [1]


def test_add_measure_by_index_sparse_then_append_fills_next_bound_count():
    """With elements {0, 2} bound, the next bare measurement takes index 2 and
    replaces it, matching the flat-index bookkeeping it stands in for."""
    circuit = Circuit()
    circuit.add_measure_by_index((0, 1), [0, 2], allow_remeasure=True)
    circuit.add_measure_by_index((5,), allow_remeasure=True)
    assert circuit.anonymous_register().sources == [0, None, 5]


def test_clear_measurement():
    circuit = Circuit()
    c = circuit.declare_register("c", 2)
    circuit.add_measure((0, 1), c)
    circuit.clear_measurement(c, 0)
    assert c.sources == [None, 1]
    assert circuit.measurement_slots == [(c, 1, 1)]
    # the qubit stays part of the circuit
    assert circuit.qubit_set == {0, 1}


def test_measurement_slots_follow_declaration_then_element_order():
    circuit = Circuit()
    c = circuit.declare_register("c", 2)
    d = circuit.declare_register("d", 2)
    circuit.add_measure((1,), d, [1])
    circuit.add_measure((0,), c, [1])
    assert circuit.measurement_slots == [(c, 1, 0), (d, 1, 1)]
    assert circuit.measured_qubits == [0, 1]


def test_classical_register_repr():
    register = ClassicalRegister("c", 2, 0)
    register.bind(1, 4)
    assert repr(register) == "ClassicalRegister(name='c', size=2, sources=[None, 4])"
