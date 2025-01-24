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

from braket.analog_hamiltonian_simulator.rydberg.validators.times_series import TimeSeriesValidator


@pytest.fixture
def time_series_data():
    return {
        "times": [0.0, 1e-8, 2e-8, 3e-8],
        "values": [0.0, 0.1, 0.2, 0.3],
    }


def test_time_series(time_series_data, device_capabilities_constants):
    try:
        TimeSeriesValidator(capabilities=device_capabilities_constants, **time_series_data)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "times, error_message",
    [
        ([], "Length of times must be at least 2; it is 0"),
        ([0.0], "Length of times must be at least 2; it is 1"),
    ],
)
# Rule: There must be at least 2 times for any component of the effective Hamiltonian
def test_time_series_times_at_least_2_timepoints(
    times, error_message, device_capabilities_constants
):
    data = {"times": times}
    _assert_time_series_fields(data, error_message, device_capabilities_constants)


@pytest.mark.parametrize(
    "times, error_message",
    [
        (
            [1.0e-6, 3.0e-6],
            "First time value is 1e-06; it must be 0.0",
        ),
        (
            [-1.0e-6, 3.0e-6],
            "First time value is -1e-06; it must be 0.0",
        ),
    ],
)
# Rule: The times for any component of the effective Hamiltonian must start at 0
def test_time_series_times_start_with_0(times, error_message, device_capabilities_constants):
    data = {"times": times, "values": [0.0, 0.1]}
    _assert_time_series_fields(data, error_message, device_capabilities_constants)


@pytest.mark.parametrize(
    "times, warning_message",
    [
        (
            [0.0, 25.0e-6],
            "Max time is 2.5e-05 seconds which is bigger than the typical scale (0.000020 seconds). "
            "The time points should  be specified in SI units.",
        ),
        (
            [0.0, 0.000025],
            "Max time is 2.5e-05 seconds which is bigger than the typical scale (0.000020 seconds). "
            "The time points should  be specified in SI units.",
        ),
    ],
)
#  Rule: The times for any component of the effective Hamiltonian must remain in bounds.
def test_time_series_times_are_not_too_big(times, warning_message, device_capabilities_constants):
    data = {"times": times, "values": [0.0, 0.1]}
    _assert_time_series_fields_2(data, warning_message, device_capabilities_constants)


@pytest.mark.parametrize(
    "times, values, error_message",
    [
        (
            [0.0, 4.0e-6, 2.0e-6],
            [0.0, 0.1, 0.2],
            "Time point 1 (4e-06) and time point 2 (2e-06) must be monotonically increasing.",
        ),
        (
            [0.0, 4.0e-6, 2.0e-6, 1.0e-6],
            [0.0, 0.1, 0.2, 0.3],
            "Time point 1 (4e-06) and time point 2 (2e-06) must be monotonically increasing.",
        ),
        (
            [0.0, 4.0e-6, 4.0e-6],
            [0.0, 0.1, 0.2],
            "Time point 1 (4e-06) and time point 2 (4e-06) must be monotonically increasing.",
        ),
    ],
)
# Rule: The times for any component of the effective Hamiltonian must be sorted in strict
#       ascending order
def test_time_series_times_must_be_ascendingly_sorted(
    times, values, error_message, device_capabilities_constants
):
    data = {"times": times, "values": values}
    _assert_time_series_fields(data, error_message, device_capabilities_constants)


@pytest.mark.parametrize(
    "values, times, error_message",
    [
        (
            [0.0, 2.5e7, 2.5e7, 0.0],
            [0, 2.7e-6, 3e-6],
            " The sample times (length: 3) and the values (length: 4) must have the same length.",
        )
    ],
)
# Rule: The global Rydberg Rabi frequency amplitude must contain the same number of times and
#       values.
# Rule: The global Rydberg Rabi frequency phase must contain the same number of times and values.
# Rule: The global Rydberg detuning must contain the same number of times and values.
def test_driving_field_times_and_values_have_same_length(
    values, times, error_message, device_capabilities_constants
):
    data = {"times": times, "values": values}
    _assert_time_series_fields(data, error_message, device_capabilities_constants)


def _assert_time_series_fields(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        TimeSeriesValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)


def _assert_time_series_fields_2(data, warning_message, device_capabilities_constants):
    with pytest.warns(UserWarning) as e:
        TimeSeriesValidator(capabilities=device_capabilities_constants, **data)
    assert warning_message in str(e[-1].message)
