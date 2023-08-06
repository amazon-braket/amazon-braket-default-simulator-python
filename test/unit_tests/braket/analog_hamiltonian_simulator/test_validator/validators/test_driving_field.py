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
from braket.ir.ahs.driving_field import DrivingField
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.driving_field import (
    DrivingFieldValidator,
)


@pytest.fixture
def driving_field_data():
    return {
        "amplitude": {
            "pattern": "uniform",
            "time_series": {
                "times": [0, 1e-07, 3.9e-06, 4e-06],
                "values": [0, 12566400.0, 12566400.0, 0],
            },
        },
        "phase": {
            "pattern": "uniform",
            "time_series": {
                "times": [0, 1e-07, 3.9e-06, 4e-06],
                "values": [0, 0, -16.0832, -16.0832],
            },
        },
        "detuning": {
            "pattern": "uniform",
            "time_series": {
                "times": [0, 1e-07, 3.9e-06, 4e-06],
                "values": [-125000000, -125000000, 125000000, 125000000],
            },
        },
    }


@pytest.fixture
def mock_driving_field_data():
    data = {
        "amplitude": {
            "pattern": "uniform",
            "time_series": {
                "times": [],
                "values": [],
            },
        },
        "phase": {
            "pattern": "uniform",
            "time_series": {
                "times": [],
                "values": [],
            },
        },
        "detuning": {
            "pattern": "uniform",
            "time_series": {
                "times": [],
                "values": [],
            },
        },
    }
    return DrivingField.parse_obj(data).dict()


def test_driving_field(driving_field_data, device_capabilities_constants):
    try:
        DrivingFieldValidator(capabilities=device_capabilities_constants, **driving_field_data)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "amplitude_times, phase_times, detuning_times, error_message",
    [
        (
            [0, 1e-07, 3.9e-06, 3e-06],
            [0, 1e-07, 3.9e-06, 4e-06],
            [0, 1e-07, 3.9e-06, 4e-06],
            "The last timepoints for all the sequences are not equal. They are",
        )
    ],
)
# Rule: The last value in the `times` array for each component of the effective
#       Hamiltonian must all be equal.
def test_driving_field_sequences_have_the_same_end_time(
    amplitude_times,
    phase_times,
    detuning_times,
    error_message,
    mock_driving_field_data,
    device_capabilities_constants,
):
    mock_driving_field_data["amplitude"]["time_series"]["times"] = amplitude_times
    mock_driving_field_data["phase"]["time_series"]["times"] = phase_times
    mock_driving_field_data["detuning"]["time_series"]["times"] = detuning_times
    _assert_driving_field(mock_driving_field_data, error_message, device_capabilities_constants)


@pytest.mark.parametrize(
    "values, field_name, warning_message",
    [
        (
            [0.0, 2.6e7, 2.5e7, 0.0],
            "amplitude",
            "Value 1 (26000000.0) in amplitude time series outside "
            "the typical range [0, 25000000.0]. "
            "The values should  be specified in SI units.",
        ),
        (
            [0.0, -0.5e7, 0.5e7, 0.0],
            "amplitude",
            "Value 1 (-5000000.0) in amplitude time series outside "
            "the typical range [0, 25000000.0]. "
            "The values should  be specified in SI units.",
        ),
        (
            [0.0, -0.5e7, 2.6e7, 0.0],
            "amplitude",
            "Value 1 (-5000000.0) in amplitude time series outside "
            "the typical range [0, 25000000.0]. "
            "The values should  be specified in SI units.",
        ),
        (
            [1.26e8],
            "detuning",
            "Value 0 (126000000.0) in detuning time series outside the typical range "
            "[-125000000.0, 125000000.0]. The values should  be specified in SI units.",
        ),
        (
            [-2e8],
            "detuning",
            "Value 0 (-200000000.0) in detuning time series outside the typical range "
            "[-125000000.0, 125000000.0]. The values should  be specified in SI units.",
        ),
    ],
)
# Rule: The global Rydberg Rabi frequency amplitude should remain in bounds.
# Rule: The global Rydberg detuning should remain in bounds.
#
# If not, warning message will be issued to remind that the frequency or detuning need to
# be specified with SI units.
def test_driving_field_values_within_range(
    values, field_name, warning_message, mock_driving_field_data, device_capabilities_constants
):
    mock_driving_field_data[field_name]["time_series"]["values"] = values
    _assert_driving_field_warning_message(
        mock_driving_field_data, warning_message, device_capabilities_constants
    )


@pytest.mark.parametrize(
    "field_name, error_message",
    [
        ("amplitude", 'Pattern of amplitude must be "uniform"'),
        ("phase", 'Pattern of phase must be "uniform"'),
        ("detuning", 'Pattern of detuning must be "uniform"'),
    ],
)
def test_driving_field_pattern_is_uniform(
    field_name, error_message, mock_driving_field_data, device_capabilities_constants
):
    mock_driving_field_data[field_name]["pattern"] = "test"
    _assert_driving_field(mock_driving_field_data, error_message, device_capabilities_constants)


def test_driving_field_empty(mock_driving_field_data, device_capabilities_constants):
    try:
        DrivingFieldValidator(capabilities=device_capabilities_constants, **mock_driving_field_data)
    except ValidationError as e:
        pytest.fail(f"Validate driving field empty test is failing : {str(e)}")


def _assert_driving_field(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        DrivingFieldValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)


def _assert_driving_field_warning_message(data, warning_message, device_capabilities_constants):
    with pytest.warns(UserWarning) as e:
        DrivingFieldValidator(capabilities=device_capabilities_constants, **data)
    assert warning_message in str(e[-1].message)
