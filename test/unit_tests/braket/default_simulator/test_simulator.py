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

import numpy as np
import pytest

from braket.default_simulator import observables
from braket.default_simulator.result_types import Expectation, Variance
from braket.default_simulator.simulator import BaseLocalSimulator
from braket.simulator import BraketSimulator


@pytest.mark.parametrize(
    "obs1,obs2",
    [
        (observables.PauliX([1]), observables.PauliX(None)),
        (observables.PauliZ([1]), observables.PauliZ(None)),
        (observables.Hermitian(np.eye(2), [1]), observables.Hermitian(np.eye(2), None)),
    ],
)
def test_validate_and_consolidate_observable_result_types_none(obs1, obs2):
    obs_rts = [
        Expectation(obs1),
        Variance(obs2),
    ]
    actual_obs = BaseLocalSimulator._validate_and_consolidate_observable_result_types(obs_rts, 2)
    assert len(actual_obs) == 1
    assert actual_obs[0].measured_qubits is None


@pytest.mark.parametrize(
    "obs",
    [(observables.PauliX([1])), (observables.PauliZ([1])), (observables.Hermitian(np.eye(2), [1]))],
)
def test_validate_and_consolidate_observable_result_types_same_target(obs):
    obs_rts = [
        Expectation(obs),
        Variance(obs),
    ]
    actual_obs = BaseLocalSimulator._validate_and_consolidate_observable_result_types(obs_rts, 2)
    assert len(actual_obs) == 1
    assert actual_obs[0].measured_qubits == (1,)


def test_validate_and_consolidate_observable_result_types_tensor_product():
    obs_rts = [
        Expectation(observables.TensorProduct([observables.PauliX([0]), observables.PauliY([1])])),
        Variance(observables.TensorProduct([observables.PauliX([0]), observables.PauliY([1])])),
        Expectation(observables.TensorProduct([observables.PauliX([2]), observables.PauliY([3])])),
    ]
    actual_obs = BaseLocalSimulator._validate_and_consolidate_observable_result_types(obs_rts, 4)
    assert len(actual_obs) == 2
    assert actual_obs[0].measured_qubits == (
        0,
        1,
    )
    assert actual_obs[1].measured_qubits == (
        2,
        3,
    )


@pytest.mark.parametrize(
    "obs1,obs2",
    [
        (observables.PauliX([1]), observables.PauliX([2])),
        (observables.PauliZ([1]), observables.PauliZ([2])),
        (observables.Hermitian(np.eye(2), [1]), observables.Hermitian(np.eye(2), [2])),
    ],
)
def test_validate_and_consolidate_observable_result_types_targets(obs1, obs2):
    obs_rts = [
        Expectation(obs1),
        Expectation(obs2),
    ]
    actual_obs = BaseLocalSimulator._validate_and_consolidate_observable_result_types(obs_rts, 3)
    assert len(actual_obs) == 2
    assert actual_obs[0].measured_qubits == (1,)
    assert actual_obs[1].measured_qubits == (2,)


def test_base_local_simulator_instance_braket_simulator():
    assert isinstance(BaseLocalSimulator(), BraketSimulator)


@pytest.mark.xfail(raises=NotImplementedError)
def test_base_local_simulator_properties():
    BaseLocalSimulator().properties


@pytest.mark.xfail(raises=NotImplementedError)
def test_base_local_simulator_simulation_type():
    BaseLocalSimulator().simulation_type
