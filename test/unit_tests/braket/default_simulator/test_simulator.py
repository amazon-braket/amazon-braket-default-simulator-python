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

import numpy as np
import pytest

from braket.default_simulator import observables
from braket.default_simulator.jaqcd_simulator import BaseLocalJaqcdSimulator
from braket.default_simulator.result_types import DensityMatrix, Expectation, Probability, Variance


@pytest.mark.parametrize(
    "result_type",
    [
        Expectation(observables.PauliX([1])),
        Variance(observables.TensorProduct([observables.PauliY([0]), observables.PauliZ([1])])),
        Expectation(
            observables.TensorProduct(
                [observables.Identity([0]), observables.Hermitian(np.eye(2), [1])]
            )
        ),
        Expectation(observables.Hermitian(np.eye(4), [0, 1])),
        Variance(observables.PauliX()),
        DensityMatrix([1]),
        Probability([1]),
    ],
)
def test_validate_result_types_qubits_exist(result_type):
    BaseLocalJaqcdSimulator._validate_result_types_qubits_exist([result_type], 2)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "result_type", [Expectation(observables.PauliX([1])), DensityMatrix([1]), Probability([1])]
)
def test_validate_result_types_qubits_exist_error(result_type):
    BaseLocalJaqcdSimulator._validate_result_types_qubits_exist([result_type], 1)


def test_observable_hash_tensor_product():
    matrix = np.eye(4)
    obs = observables.TensorProduct(
        [observables.PauliX([0]), observables.Hermitian(matrix, [1, 2]), observables.PauliY([1])]
    )
    hash_dict = BaseLocalJaqcdSimulator._observable_hash(obs)
    matrix_hash = hash_dict[1]
    assert hash_dict == {0: "PauliX", 1: matrix_hash, 2: matrix_hash, 3: "PauliY"}


def test_base_local_simulator_abstract():
    abstract_methods = (
        "Can't instantiate abstract class BaseLocalJaqcdSimulator with "
        "abstract methods initialize_simulation, properties"
    )
    with pytest.raises(TypeError, match=abstract_methods):
        BaseLocalJaqcdSimulator()
