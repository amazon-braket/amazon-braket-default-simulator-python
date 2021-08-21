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

from collections import Counter

import numpy as np
import pytest

from braket.default_simulator import gate_operations, observables
from braket.default_simulator.state_vector_simulation import StateVectorSimulation

evolve_testdata = [
    ([gate_operations.Hadamard([0])], 1, [0.70710678, 0.70710678], [0.5, 0.5]),
    ([gate_operations.PauliX([0])], 1, [0, 1], [0, 1]),
    ([gate_operations.PauliY([0])], 1, [0, 1j], [0, 1]),
    ([gate_operations.PauliX([0]), gate_operations.PauliZ([0])], 1, [0, -1], [0, 1]),
    ([gate_operations.PauliX([0]), gate_operations.CX([0, 1])], 2, [0, 0, 0, 1], [0, 0, 0, 1]),
    ([gate_operations.PauliX([0]), gate_operations.CY([0, 1])], 2, [0, 0, 0, 1j], [0, 0, 0, 1]),
    ([gate_operations.PauliX([0]), gate_operations.CZ([0, 1])], 2, [0, 0, 1, 0], [0, 0, 1, 0]),
    ([gate_operations.PauliX([0]), gate_operations.Swap([0, 1])], 2, [0, 1, 0, 0], [0, 1, 0, 0]),
    ([gate_operations.PauliX([0]), gate_operations.ISwap([0, 1])], 2, [0, 1j, 0, 0], [0, 1, 0, 0]),
    (
        [gate_operations.PauliX([0]), gate_operations.Swap([0, 2])],
        3,
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ),
    ([gate_operations.PauliX([0]), gate_operations.S([0])], 1, [0, 1j], [0, 1]),
    ([gate_operations.PauliX([0]), gate_operations.Si([0])], 1, [0, -1j], [0, 1]),
    (
        [gate_operations.PauliX([0]), gate_operations.T([0])],
        1,
        [0, 0.70710678 + 0.70710678j],
        [0, 1],
    ),
    (
        [gate_operations.PauliX([0]), gate_operations.Ti([0])],
        1,
        [0, 0.70710678 - 0.70710678j],
        [0, 1],
    ),
    (
        [gate_operations.V([0])],
        1,
        [0.5 + 0.5j, 0.5 - 0.5j],
        [0.5, 0.5],
    ),
    (
        [gate_operations.Vi([0])],
        1,
        [0.5 - 0.5j, 0.5 + 0.5j],
        [0.5, 0.5],
    ),
    ([gate_operations.Identity([0])], 1, [1, 0], [1, 0]),
    ([gate_operations.Unitary([0], [[0, 1], [1, 0]])], 1, [0, 1], [0, 1]),
    (
        [gate_operations.PauliX([0]), gate_operations.PhaseShift([0], 0.15)],
        1,
        [0, 0.98877108 + 0.14943813j],
        [0, 1],
    ),
    (
        [
            gate_operations.PauliX([0]),
            gate_operations.PauliX([1]),
            gate_operations.CPhaseShift([0, 1], 0.15),
        ],
        2,
        [0, 0, 0, 0.98877108 + 0.14943813j],
        [0, 0, 0, 1],
    ),
    (
        [gate_operations.CPhaseShift00([0, 1], 0.15)],
        2,
        [0.98877108 + 0.14943813j, 0, 0, 0],
        [1, 0, 0, 0],
    ),
    (
        [gate_operations.PauliX([1]), gate_operations.CPhaseShift01([0, 1], 0.15)],
        2,
        [0, 0.98877108 + 0.14943813j, 0, 0],
        [0, 1, 0, 0],
    ),
    (
        [gate_operations.PauliX([0]), gate_operations.CPhaseShift10([0, 1], 0.15)],
        2,
        [0, 0, 0.98877108 + 0.14943813j, 0],
        [0, 0, 1, 0],
    ),
    ([gate_operations.RotX([0], 0.15)], 1, [0.99718882, -0.07492971j], [0.99438554, 0.00561446]),
    (
        [gate_operations.PauliX([0]), gate_operations.RotY([0], 0.15)],
        1,
        [-0.07492971, 0.99718882],
        [0.00561446, 0.99438554],
    ),
    (
        [gate_operations.Hadamard([0]), gate_operations.RotZ([0], 0.15)],
        1,
        [0.70511898 - 0.0529833j, 0.70511898 + 0.0529833j],
        [0.5, 0.5],
    ),
    (
        [gate_operations.PauliX([0]), gate_operations.PSwap([0, 1], 0.15)],
        2,
        [0, 0.98877108 + 0.14943813j, 0, 0],
        [0, 1, 0, 0],
    ),
    (
        [gate_operations.PauliX([0]), gate_operations.XY([0, 1], 0.15)],
        2,
        [0, 0.07492971j, 0.99718882, 0],
        [0, 0.00561446, 0.99438554, 0],
    ),
    (
        [gate_operations.XX([0, 1], 0.3)],
        2,
        [0.98877108, 0, 0, -0.14943813j],
        [0.97766824, 0, 0, 0.02233176],
    ),
    (
        [gate_operations.YY([0, 1], 0.3)],
        2,
        [0.98877108, 0, 0, 0.14943813j],
        [0.97766824, 0, 0, 0.02233176],
    ),
    ([gate_operations.ZZ([0, 1], 0.15)], 2, [0.99718882 - 0.07492971j, 0, 0, 0], [1, 0, 0, 0]),
    (
        [
            gate_operations.PauliX([0]),
            gate_operations.PauliX([1]),
            gate_operations.CCNot([0, 1, 2]),
        ],
        3,
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ),
    (
        [
            gate_operations.PauliX([0]),
            gate_operations.PauliX([1]),
            gate_operations.CSwap([0, 1, 2]),
        ],
        3,
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ),
]

apply_observables_testdata = [
    ([observables.PauliX([0])], [gate_operations.Hadamard([0])], 1),
    ([observables.PauliZ([0])], [], 1),
    ([observables.Identity([0])], [], 1),
    (
        [observables.PauliX([0]), observables.PauliZ([3]), observables.Hadamard([2])],
        [gate_operations.Hadamard([0]), gate_operations.RotY([2], -np.pi / 4)],
        5,
    ),
    (
        [
            observables.TensorProduct(
                [
                    observables.PauliX([0]),
                    observables.PauliZ([3]),
                    observables.Hadamard([2]),
                    observables.Identity([1]),
                ]
            )
        ],
        [gate_operations.Hadamard([0]), gate_operations.RotY([2], -np.pi / 4)],
        5,
    ),
    ([observables.PauliX()], [gate_operations.Hadamard([0]), gate_operations.Hadamard([1])], 2),
    ([observables.PauliZ()], [], 2),
    ([observables.Identity()], [], 2),
    ([observables.TensorProduct([observables.Identity([2]), observables.PauliZ([0])])], [], 3),
    (
        [observables.TensorProduct([observables.PauliX([2]), observables.PauliZ([0])])],
        [gate_operations.Hadamard([2])],
        3,
    ),
]


@pytest.fixture
def qft_circuit_operations():
    def _qft_operations(qubit_count):
        qft_ops = []
        for target_qubit in range(qubit_count):
            angle = np.pi / 2
            qft_ops.append(gate_operations.Hadamard([target_qubit]))
            for control_qubit in range(target_qubit + 1, qubit_count):
                qft_ops.append(gate_operations.CPhaseShift([control_qubit, target_qubit], angle))
                angle /= 2
        return qft_ops

    return _qft_operations


@pytest.mark.parametrize(
    "instructions, qubit_count, state_vector, probability_amplitudes", evolve_testdata
)
def test_simulation_simple_circuits(
    instructions, qubit_count, state_vector, probability_amplitudes
):
    # +2 to ensure evolve doesn't fail for partition size greater than
    # or equal to the number of instructions
    for batch_size in range(1, len(instructions) + 2):
        simulation = StateVectorSimulation(qubit_count, 0, batch_size)
        simulation.evolve(instructions)
        assert np.allclose(state_vector, simulation.state_vector)
        assert np.allclose(probability_amplitudes, simulation.probabilities)


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("obs, equivalent_gates, qubit_count", apply_observables_testdata)
def test_apply_observables(obs, equivalent_gates, qubit_count, batch_size):
    sim_observables = StateVectorSimulation(qubit_count, 0, batch_size)
    sim_observables.apply_observables(obs)
    sim_gates = StateVectorSimulation(qubit_count, 0, batch_size)
    sim_gates.evolve(equivalent_gates)
    assert np.allclose(sim_observables.state_with_observables, sim_gates.state_vector)


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.xfail(raises=RuntimeError)
def test_apply_observables_fails_second_call(batch_size):
    simulation = StateVectorSimulation(4, 0, batch_size)
    simulation.apply_observables([observables.PauliX([0])])
    simulation.apply_observables([observables.PauliX([0])])


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.xfail(raises=RuntimeError)
def test_state_with_observables_fails_before_applying(batch_size):
    StateVectorSimulation(4, 0, batch_size).state_with_observables


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_simulation_qft_circuit(qft_circuit_operations, batch_size):
    qubit_count = 16
    simulation = StateVectorSimulation(qubit_count, 0, batch_size)
    operations = qft_circuit_operations(qubit_count)
    simulation.evolve(operations)
    assert np.allclose(simulation.probabilities, [1 / (2 ** qubit_count)] * (2 ** qubit_count))


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_simulation_retrieve_samples(batch_size):
    simulation = StateVectorSimulation(2, 10000, batch_size)
    simulation.evolve([gate_operations.Hadamard([0]), gate_operations.CX([0, 1])])
    counter = Counter(simulation.retrieve_samples())
    assert simulation.qubit_count == 2
    assert counter.keys() == {0, 3}
    assert 0.4 < counter[0] / (counter[0] + counter[3]) < 0.6
    assert 0.4 < counter[3] / (counter[0] + counter[3]) < 0.6
    assert counter[0] + counter[3] == 10000


@pytest.mark.parametrize("batch_size", [None, 3.14, "foo"])
@pytest.mark.xfail(raises=TypeError)
def test_instantiation_fails_batch_size_wrong_type(batch_size):
    StateVectorSimulation(4, 0, batch_size)


@pytest.mark.parametrize("batch_size", [0, -5])
@pytest.mark.xfail(raises=ValueError)
def test_instantiation_fails_batch_size_nonpositive(batch_size):
    StateVectorSimulation(4, 0, batch_size)
