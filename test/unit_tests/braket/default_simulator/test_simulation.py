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

from braket.default_simulator.simulation import Simulation

test_data = [(5, 100), (8, 0), (1, 1000), (1, 0)]


@pytest.fixture
def simulation():
    return Simulation(qubit_count=1, shots=0)


@pytest.mark.parametrize("qubit_count, shots", test_data)
def test_simulation(qubit_count, shots):
    sim = Simulation(qubit_count, shots)
    assert sim.qubit_count == qubit_count
    assert sim.shots == shots


@pytest.mark.xfail(raises=NotImplementedError)
def test_simulation_evolve(simulation):
    simulation.evolve([])


@pytest.mark.xfail(raises=NotImplementedError)
def test_simulation_expectation(simulation):
    simulation.expectation(None)


@pytest.mark.xfail(raises=NotImplementedError)
def test_simulation_retrieve_samples(simulation):
    simulation.retrieve_samples()


@pytest.mark.xfail(raises=NotImplementedError)
def test_simulation_probabilities(simulation):
    simulation.probabilities()
