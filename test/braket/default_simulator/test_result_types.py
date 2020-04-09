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
from braket.default_simulator.observables import Hadamard, PauliX, TensorProduct
from braket.default_simulator.result_types import (
    Amplitude,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)


@pytest.fixture
def state_vector():
    multiplier = np.concatenate([np.ones(8), np.ones(8) * 1j])
    return ((np.arange(16) / 120) ** (1 / 2)) * multiplier


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
    return TensorProduct([PauliX([1]), Hadamard([2])])


def test_state_vector(state_vector):
    assert np.allclose(StateVector().calculate(state_vector), state_vector)


def test_amplitude(state_vector):
    amplitudes = Amplitude(["0010", "0101", "1110"]).calculate(state_vector)
    assert cmath.isclose(amplitudes["0010"], state_vector[2])
    assert cmath.isclose(amplitudes["0101"], state_vector[5])
    assert cmath.isclose(amplitudes["1110"], state_vector[14])


def test_probability(state_vector, marginal_12):
    probability_12 = Probability([1, 2]).calculate(state_vector)
    assert np.allclose(probability_12, marginal_12)

    probability_all = Probability([0, 1, 2, 3]).calculate(state_vector)
    assert np.allclose(probability_all, np.abs(state_vector) ** 2)


def test_expectation(state_vector, observable, marginal_12):
    expectation = Expectation(observable).calculate(state_vector)
    assert np.allclose(expectation, marginal_12 @ np.array([1, -1, -1, 1]))


def test_variance(state_vector, observable, marginal_12):
    variance = Variance(observable).calculate(state_vector)
    expected_eigenvalues = np.array([1, -1, -1, 1])
    expected_variance = (
        marginal_12 @ (expected_eigenvalues ** 2) - (marginal_12 @ expected_eigenvalues).real ** 2
    )
    assert np.allclose(variance, expected_variance)


def test_sample(state_vector, observable):
    shots = 1000
    sample = Sample(observable, shots).calculate(state_vector)
    assert len(sample) == shots

    # sample contains only 1 and -1 as entries
    as_binary = (sample + 1) / 2
    assert np.array_equal(as_binary, as_binary.astype(bool))
