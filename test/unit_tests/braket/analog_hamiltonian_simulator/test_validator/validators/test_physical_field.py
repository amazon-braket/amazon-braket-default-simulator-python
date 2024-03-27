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
from pydantic.v1.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.physical_field import (
    PhysicalFieldValidator,
)


@pytest.fixture
def physical_field_data_1():
    return {
        "time_series": {
            "times": [0.0, 1e-8, 2e-8, 3e-8],
            "values": [0.0, 0.1, 0.2, 0.3],
        },
        "pattern": "uniform",
    }


@pytest.fixture
def physical_field_data_2():
    return {
        "time_series": {
            "times": [0.0, 1e-8, 2e-8, 3e-8],
            "values": [0.0, 0.1, 0.2, 0.3],
        },
        "pattern": [0.0, 0.1, 0.2, 0.3],
    }


def test_physical_field(physical_field_data_1, physical_field_data_2):
    try:
        PhysicalFieldValidator(**physical_field_data_1)
        PhysicalFieldValidator(**physical_field_data_2)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


def test_physical_field_pattern_str(physical_field_data_1):
    physical_field_data_1["pattern"] = "test"
    with pytest.raises(ValidationError) as e:
        PhysicalFieldValidator(**physical_field_data_1)
    assert 'Invalid pattern string (test); only string: "uniform"' in str(e.value)
