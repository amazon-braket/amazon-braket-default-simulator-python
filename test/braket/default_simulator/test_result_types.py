# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from braket.default_simulator import StateVectorSimulation
from braket.default_simulator.observables import Hadamard, PauliX, TensorProduct
from braket.default_simulator.result_types import (
    Amplitude,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
    from_braket_result_type,
)
from braket.ir import jaqcd
from braket.ir.jaqcd import shared_models

NUM_SAMPLES = 1000

observable_type_testdata = [
    (
        jaqcd.Expectation(targets=[1], observable=["x"]),
        {"type": "expectation", "operator": "x", "targets": [1]},
    ),
    (
        jaqcd.Variance(targets=[1], observable=["y"]),
        {"type": "variance", "operator": "y", "targets": [1]},
    ),
    (
        jaqcd.Sample(targets=[1], observable=["z"]),
        {"type": "sample", "operator": "z", "targets": [1]},
    ),
    (
        jaqcd.Expectation(observable=["h"]),
        {"type": "expectation", "operator": "h", "targets": None},
    ),
    (
        jaqcd.Variance(targets=[0], observable=[[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]]),
        {
            "type": "variance",
            "operator": "hermitian([[0.+0.j,1.+0.j],[1.+0.j,0.+0.j]])",
            "targets": [0],
        },
    ),
    (
        jaqcd.Sample(observable=[[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]]),
        {
            "type": "sample",
            "operator": "hermitian([[0.+0.j,1.+0.j],[1.+0.j,0.+0.j]])",
            "targets": None,
        },
    ),
    (
        jaqcd.Variance(targets=[0, 1], observable=["h", "i"]),
        {"type": "variance", "operator": "tensorproduct(h,i)", "targets": [0, 1]},
    ),
]


@pytest.fixture
def state_vector():
    multiplier = np.concatenate([np.ones(8), np.ones(8) * 1j])
    return ((np.arange(16) / 120) ** (1 / 2)) * multiplier


@pytest.fixture
def simulation(state_vector):
    sim = StateVectorSimulation(4, NUM_SAMPLES)
    sim._state_vector = state_vector
    sim._post_observables = state_vector  # Same for simplicity
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


@pytest.fixture
def observable():
    return TensorProduct([PauliX(constituent=True), Hadamard(constituent=True)], [1, 2])


def test_state_vector(simulation, state_vector):
    result_type = StateVector()
    assert np.allclose(result_type.calculate(simulation), state_vector)
    assert result_type.result_info == {"type": "state_vector"}


def test_amplitude(simulation, state_vector):
    result_type = Amplitude(["0010", "0101", "1110"])
    amplitudes = result_type.calculate(simulation)
    assert cmath.isclose(amplitudes["0010"], state_vector[2])
    assert cmath.isclose(amplitudes["0101"], state_vector[5])
    assert cmath.isclose(amplitudes["1110"], state_vector[14])
    assert result_type.result_info == {"type": "amplitude", "states": ["0010", "0101", "1110"]}


def test_probability(simulation, state_vector, marginal_12):
    result_type = Probability([1, 2])
    probability_12 = Probability([1, 2]).calculate(simulation)
    assert np.allclose(probability_12, marginal_12)
    assert result_type.result_info == {"type": "probability", "targets": [1, 2]}

    state_vector_probabilities = np.abs(state_vector) ** 2

    probability_no_targets = Probability().calculate(simulation)
    assert np.allclose(probability_no_targets, state_vector_probabilities)

    probability_all_qubits = Probability([0, 1, 2, 3]).calculate(simulation)
    assert np.allclose(probability_all_qubits, state_vector_probabilities)


def test_expectation(simulation, observable, marginal_12):
    expectation = Expectation(observable).calculate(simulation)
    assert np.allclose(expectation, marginal_12 @ np.array([1, -1, -1, 1]))


def test_expectation_no_targets():
    simulation = StateVectorSimulation(2)
    simulation._post_observables = np.array([1, 1, 0, 0]) / np.sqrt(2)
    expectation = Expectation(PauliX()).calculate(simulation)
    assert np.allclose(expectation, [1, 0])


def test_variance(simulation, observable, marginal_12):
    variance = Variance(observable).calculate(simulation)
    expected_eigenvalues = np.array([1, -1, -1, 1])
    expected_variance = (
        marginal_12 @ (expected_eigenvalues ** 2) - (marginal_12 @ expected_eigenvalues).real ** 2
    )
    assert np.allclose(variance, expected_variance)


def test_variance_no_targets():
    simulation = StateVectorSimulation(2)
    simulation._post_observables = np.array([1, 0, 0, 1]) / np.sqrt(2)
    variance = Variance(PauliX()).calculate(simulation)
    assert np.allclose(variance, [1, 1])


def test_sample(simulation, observable):
    sample = Sample(observable).calculate(simulation)
    assert len(sample) == NUM_SAMPLES

    # sample contains only 1 and -1 as entries
    assert all([x in {-1, 1} for x in sample])


def test_sample_no_targets():
    simulation = StateVectorSimulation(2, NUM_SAMPLES)
    simulation._post_observables = np.array([1, 0, 0, 1]) / np.sqrt(2)
    sample = Sample(PauliX()).calculate(simulation)
    assert len(sample) == 2
    for qubit_sample in sample:
        assert len(qubit_sample) == NUM_SAMPLES
        assert all([x in {-1, 1} for x in qubit_sample])


def test_from_braket_result_type_statevector():
    assert isinstance(from_braket_result_type(jaqcd.StateVector()), StateVector)


def test_from_braket_result_type_amplitude():
    translated = from_braket_result_type(jaqcd.Amplitude(states=["01", "10"]))
    assert isinstance(translated, Amplitude)
    assert translated._states == ["01", "10"]


def test_from_braket_result_type_probability():
    translated = from_braket_result_type(jaqcd.Probability(targets=[0, 1]))
    assert isinstance(translated, Probability)
    assert translated._targets == [0, 1]


@pytest.mark.parametrize("braket_result_type, result_info", observable_type_testdata)
def test_from_braket_result_type_observable(braket_result_type, result_info):
    assert from_braket_result_type(braket_result_type).result_info == result_info


@pytest.mark.xfail(raises=ValueError)
def test_from_braket_result_type_unsupported_type():
    from_braket_result_type(shared_models.OptionalMultiTarget(targets=[4, 3]))


@pytest.mark.xfail(raises=ValueError)
def test_from_braket_result_type_unknown_observable():
    from_braket_result_type(
        jaqcd.Variance(targets=[0], observable=[[[[0, 0], [1, 0], [3, 2]], [[1, 0], [0, 0]]]])
    )
