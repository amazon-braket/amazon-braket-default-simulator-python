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


def test_anonymous_register_is_placed_at_first_use():
    circuit = Circuit()
    c = circuit.declare_register("c", 1)
    circuit.add_measure((3,))
    d = circuit.declare_register("d", 1)
    anonymous = circuit.anonymous_register()
    assert circuit.classical_registers == [c, anonymous, d]


def test_measured_registers_empty_when_nothing_measured():
    circuit = Circuit()
    circuit.declare_register("c", 2)
    assert circuit.measured_registers == []


def test_measured_registers_skips_unmeasured():
    circuit = Circuit()
    circuit.declare_register("unused", 2)
    c = circuit.declare_register("c", 2)
    circuit.add_measure((1,), c, [1])
    assert circuit.measured_registers == [c]
    assert circuit.measured_qubits == [1]
    circuit.validate_single_measured_register()


def test_validate_rejects_two_declared_registers():
    circuit = Circuit()
    c = circuit.declare_register("c", 2)
    d = circuit.declare_register("d", 2)
    circuit.add_measure((0,), c, [1])
    circuit.add_measure((1,), d, [1])
    with pytest.raises(ValueError, match="recorded into 2: `c`, `d`"):
        circuit.validate_single_measured_register()


def test_validate_rejects_declared_plus_anonymous():
    circuit = Circuit()
    c = circuit.declare_register("c", 1)
    circuit.add_measure((0,), c)
    circuit.add_measure((1,))
    with pytest.raises(ValueError, match="`c`, measurements without a destination"):
        circuit.validate_single_measured_register()


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
