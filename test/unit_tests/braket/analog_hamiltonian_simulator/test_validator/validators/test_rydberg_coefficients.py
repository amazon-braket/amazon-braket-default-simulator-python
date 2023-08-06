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
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.validators.rydberg_coefficient import (
    validate_rydberg_interaction_coef,
)


@pytest.fixture
def valid_rydberg_coefficient():
    return RYDBERG_INTERACTION_COEF


def test_rydberg_interaction_coef(valid_rydberg_coefficient):
    try:
        validate_rydberg_interaction_coef(valid_rydberg_coefficient)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "invalid_rydberg_coefficient, error_message",
    [(-5.4e-24, "`rydberg_interaction_coef` needs to be positive.")],
)
def test_invalid_rydberg_coefficient_error_message(invalid_rydberg_coefficient, error_message):
    with pytest.raises(ValueError) as e:
        validate_rydberg_interaction_coef(invalid_rydberg_coefficient)
    assert error_message in str(e.value)


@pytest.mark.parametrize(
    "invalid_blockade_radius, warning_message",
    [
        (
            5.4,
            "Rydberg interaction coefficient 5.4 meter^6/second is not in the same scale as "
            "the typical value (5.42e-24 meter^6/second). "
            "The Rydberg interaction coefficient should be specified in SI units.",
        )
    ],
)
def test_invalid_rydberg_coefficient_warning_message(invalid_blockade_radius, warning_message):
    with pytest.warns(UserWarning) as e:
        validate_rydberg_interaction_coef(invalid_blockade_radius)
    assert warning_message in str(e[-1].message)
