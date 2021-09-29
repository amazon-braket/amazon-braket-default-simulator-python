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

import cmath

import numpy as np
import pytest
from braket.ir import jaqcd
from braket.ir.jaqcd import shared_models

from braket.default_simulator import StateVectorSimulation, observables
from braket.default_simulator.linalg_utils import marginal_probability
from braket.default_simulator.result_types import (
    Amplitude,
    DensityMatrix,
    Expectation,
    Probability,
    StateVector,
    Variance,
    from_braket_result_type,
)

NUM_SAMPLES = 1000

observables_testdata = [
    observables.TensorProduct([observables.PauliX([1]), observables.Hadamard([2])]),
    observables.TensorProduct([observables.Identity([0]), observables.PauliY([2])]),
]

all_qubit_observables_testdata = [
    observables.PauliX(),
    observables.PauliY(),
    observables.PauliZ(),
    observables.Hadamard(),
    observables.Identity(),
    observables.Hermitian(np.array([[0, 1], [1, 0]])),
]

observable_type_testdata = [
    (jaqcd.Expectation(targets=[1], observable=["x"]), Expectation),
    (jaqcd.Variance(targets=[1], observable=["y"]), Variance),
    (jaqcd.Variance(targets=[1], observable=["z"]), Variance),
    (jaqcd.Expectation(observable=["h"]), Expectation),
    (jaqcd.Variance(targets=[0], observable=[[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]]), Variance),
    (jaqcd.Expectation(targets=[0, 1], observable=["h", "i"]), Expectation),
    (
        jaqcd.Variance(targets=[0, 1], observable=["h", [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]]),
        Variance,
    ),
]


@pytest.fixture
def state_vector():
    multiplier = np.concatenate([np.ones(8), np.ones(8) * 1j])
    return ((np.arange(16) / 120) ** (1 / 2)) * multiplier


@pytest.fixture
def density_matrix(state_vector):
    return np.outer(state_vector, state_vector.conj())


@pytest.fixture
def observable():
    return observables.TensorProduct([observables.PauliX([1]), observables.Hadamard([2])])


@pytest.fixture
def simulation(state_vector, observable):
    sim = StateVectorSimulation(4, NUM_SAMPLES, 1)
    sim._state_vector = state_vector
    sim.apply_observables([observable])
    return sim


@pytest.fixture
def marginal_12(state_vector):
    all_probs = np.abs(state_vector) ** 2
    return np.array(
        [
            np.sum(all_probs[[0, 1, 8, 9]]),
            np.sum(all_probs[[2, 3, 10, 11]]),
            np.sum(all_probs[[4, 5, 12, 13]]),
            np.sum(all_probs[[6, 7, 14, 15]]),
        ]
    )


def test_state_vector(simulation, state_vector):
    result_type = StateVector()
    assert np.allclose(result_type.calculate(simulation), state_vector)


def test_density_matrix(simulation, density_matrix):
    density_matrix_12 = DensityMatrix([3]).calculate(simulation)
    partial_trace_12 = np.array(
        [[0.46666667 + 0.0j, 0.49464257 + 0.0j], [0.49464257 + 0.0j, 0.53333333 + 0.0j]]
    )
    assert np.allclose(density_matrix_12, partial_trace_12)

    density_matrix_no_targets = DensityMatrix().calculate(simulation)
    assert np.allclose(density_matrix_no_targets, density_matrix)

    density_matrix_all_qubits = DensityMatrix([0, 1, 2, 3]).calculate(simulation)
    assert np.allclose(density_matrix_all_qubits, density_matrix)


@pytest.mark.xfail(raises=IndexError)
def test_density_matrix_wrong_targets(simulation):
    DensityMatrix([4]).calculate(simulation)


def test_amplitude(simulation, state_vector):
    result_type = Amplitude(["0010", "0101", "1110"])
    amplitudes = result_type.calculate(simulation)
    assert cmath.isclose(amplitudes["0010"], state_vector[2])
    assert cmath.isclose(amplitudes["0101"], state_vector[5])
    assert cmath.isclose(amplitudes["1110"], state_vector[14])


def test_probability(simulation, state_vector, marginal_12):
    probability_12 = Probability([1, 2]).calculate(simulation)
    assert np.allclose(probability_12, marginal_12)

    state_vector_probabilities = np.abs(state_vector) ** 2

    probability_no_targets = Probability().calculate(simulation)
    assert np.allclose(probability_no_targets, state_vector_probabilities)

    probability_all_qubits = Probability([0, 1, 2, 3]).calculate(simulation)
    assert np.allclose(probability_all_qubits, state_vector_probabilities)


def _create_simulation(state, obs):
    sim = StateVectorSimulation(4, NUM_SAMPLES, 1)
    sim._state_vector = state
    sim.apply_observables([obs])
    return sim


@pytest.mark.parametrize("obs", observables_testdata)
def test_expectation(state_vector, obs):
    simulation = _create_simulation(state_vector, obs)
    result_type = Expectation(obs)
    assert result_type.observable == obs

    calculated = result_type.calculate(simulation)
    from_diagonalization = _expectation_from_diagonalization(
        np.abs(simulation.state_with_observables) ** 2,
        obs.measured_qubits,
        obs.eigenvalues,
    )
    assert np.allclose(calculated, from_diagonalization)


@pytest.mark.parametrize("obs", all_qubit_observables_testdata)
def test_expectation_no_targets(state_vector, obs):
    simulation = _create_simulation(state_vector, obs)
    result_type = Expectation(obs)
    assert result_type.observable == obs

    calculated = result_type.calculate(simulation)
    diagonalized_probs = np.abs(simulation.state_with_observables) ** 2
    qubit_count = simulation.qubit_count
    eigs = obs.eigenvalues
    from_diagonalization = [
        _expectation_from_diagonalization(diagonalized_probs, [qubit], eigs)
        for qubit in range(qubit_count)
    ]

    assert np.allclose(calculated, from_diagonalization)


def _expectation_from_diagonalization(diagonalized_probs, qubits, eigenvalues):
    marginal = marginal_probability(diagonalized_probs, qubits)
    return (marginal @ eigenvalues).real


@pytest.mark.parametrize("obs", observables_testdata)
def test_variance(state_vector, obs):
    simulation = _create_simulation(state_vector, obs)
    result_type = Variance(obs)
    assert result_type.observable == obs

    calculated = result_type.calculate(simulation)
    from_diagonalization = _variance_from_diagonalization(
        np.abs(simulation.state_with_observables) ** 2,
        obs.measured_qubits,
        obs.eigenvalues,
    )
    assert np.allclose(calculated, from_diagonalization)


@pytest.mark.parametrize("obs", all_qubit_observables_testdata)
def test_variance_no_targets(state_vector, obs):
    simulation = _create_simulation(state_vector, obs)
    result_type = Variance(obs)
    assert result_type.observable == obs

    calculated = result_type.calculate(simulation)
    diagonalized_probs = np.abs(simulation.state_with_observables) ** 2
    qubit_count = simulation.qubit_count
    eigs = obs.eigenvalues
    from_diagonalization = [
        _variance_from_diagonalization(diagonalized_probs, [qubit], eigs)
        for qubit in range(qubit_count)
    ]

    assert np.allclose(calculated, from_diagonalization)


def _variance_from_diagonalization(diagonalized_probs, qubits, eigenvalues):
    marginal = marginal_probability(diagonalized_probs, qubits)
    return marginal @ (eigenvalues.real ** 2) - (marginal @ eigenvalues).real ** 2


def test_from_braket_result_type_statevector():
    assert isinstance(from_braket_result_type(jaqcd.StateVector()), StateVector)


def test_from_braket_result_type_densitymatrix():
    assert isinstance(from_braket_result_type(jaqcd.DensityMatrix()), DensityMatrix)


def test_from_braket_result_type_amplitude():
    translated = from_braket_result_type(jaqcd.Amplitude(states=["01", "10"]))
    assert isinstance(translated, Amplitude)
    assert translated._states == ["01", "10"]


def test_from_braket_result_type_probability():
    translated = from_braket_result_type(jaqcd.Probability(targets=[0, 1]))
    assert isinstance(translated, Probability)
    assert translated._targets == [0, 1]


@pytest.mark.parametrize("braket_result_type, result_type", observable_type_testdata)
def test_from_braket_result_type_observable(braket_result_type, result_type):
    assert isinstance(from_braket_result_type(braket_result_type), result_type)


@pytest.mark.xfail(raises=ValueError)
def test_from_braket_result_type_tensor_product_extra():
    from_braket_result_type(jaqcd.Expectation(targets=[0, 1, 2], observable=["h", "i"]))


@pytest.mark.xfail(raises=ValueError)
def test_from_braket_result_type_tensor_product_insufficient():
    from_braket_result_type(jaqcd.Variance(targets=[0], observable=["h", "i"]))


@pytest.mark.xfail(raises=ValueError)
def test_from_braket_result_type_unknown_observable():
    from_braket_result_type(
        jaqcd.Expectation(targets=[0], observable=[[[[0, 0], [1, 0], [3, 2]], [[1, 0], [0, 0]]]])
    )


@pytest.mark.xfail(raises=NotImplementedError)
def test_from_braket_result_type_unsupported_type():
    from_braket_result_type(shared_models.OptionalMultiTarget(targets=[4, 3]))
