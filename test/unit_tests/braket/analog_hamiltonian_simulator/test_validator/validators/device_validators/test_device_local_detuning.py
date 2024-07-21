import pytest
from braket.ir.ahs.local_detuning import LocalDetuning
from pydantic.v1.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators import (
    DeviceLocalDetuningValidator,
)


@pytest.fixture
def local_detuning_data():
    return {
        "magnitude": {
            "time_series": {"times": [0, 1e-07], "values": [0, 0]},
            "pattern": [0.0, 0.1, 0.1, 0.0, 0.1],
        }
    }


@pytest.fixture
def mock_local_detuning_data():
    data = {
        "magnitude": {
            "pattern": [],
            "time_series": {
                "times": [],
                "values": [],
            },
        }
    }
    return LocalDetuning.parse_obj(data).dict()


def test_valid_detuning(local_detuning_data, capabilities_with_local_rydberg):
    try:
        DeviceLocalDetuningValidator(
            capabilities=capabilities_with_local_rydberg, **local_detuning_data
        )
    except ValidationError as e:
        pytest.fail(f"Validate test is failing: {str(e.value)}")


def test_validation_no_time_series(mock_local_detuning_data, capabilities_with_local_rydberg):
    mock_local_detuning_data["magnitude"]["time_series"]["times"].clear()
    mock_local_detuning_data["magnitude"]["time_series"]["values"].clear()

    try:
        DeviceLocalDetuningValidator(
            capabilities=capabilities_with_local_rydberg, **mock_local_detuning_data
        )
    except ValidationError as e:
        pytest.fail(f"Validate test is failing: {str(e.value)}")


def test_validation_no_detuning_data(mock_local_detuning_data, non_local_capabilities_constants):
    error_message = "Local Rydberg capabilities information has not been \
            provided for local detuning."
    non_local_capabilities_constants.LOCAL_RYDBERG_CAPABILITIES = False
    with pytest.raises(ValueError) as e:
        DeviceLocalDetuningValidator(
            capabilities=non_local_capabilities_constants, **mock_local_detuning_data
        )
    assert error_message in str(e.value)


def test_shifting_field_magnitude_pattern_have_not_too_many_nonzeros(
    mock_local_detuning_data, capabilities_with_local_rydberg
):
    pattern = [0.1] * 257
    mock_local_detuning_data["magnitude"]["pattern"] = pattern
    error_message = (
        "Number of nonzero magnitude pattern values is 257; it must not be more than 200"
    )
    _assert_local_detuning(mock_local_detuning_data, error_message, capabilities_with_local_rydberg)


@pytest.mark.parametrize(
    "values, times, error_message",
    [
        (
            [0.0, 2.5e7, 0.0],
            [0.0, 0.01e-6, 2.0e-6],
            "For the magnitude field, rate of change of values\
                (between the 0-th and the 1-th times)\
                    is 2500000000000000.0, more than 1256600000000000.0",
        ),
        (
            [0.0, 2.5e7, 2.5e7, 0.0],
            [0.0, 0.01e-6, 3.2e-6, 3.21e-6],
            "For the magnitude field, rate of change of values\
                (between the 0-th and the 1-th times)\
                    is 2500000000000000.0, more than 1256600000000000.0",
        ),
    ],
)
def test_local_detuning_slopes_not_too_steep(
    values, times, error_message, mock_local_detuning_data, capabilities_with_local_rydberg
):
    mock_local_detuning_data["magnitude"]["time_series"]["values"] = values
    mock_local_detuning_data["magnitude"]["time_series"]["times"] = times
    _assert_local_detuning(mock_local_detuning_data, error_message, capabilities_with_local_rydberg)


@pytest.mark.parametrize(
    "times, error_message",
    [
        (
            [0.0, 12.1e-9],
            "time point 1 (1.21E-8) of magnitude time_series is\
                defined with too many digits; it must be an integer multiple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 4e-6],
            "time point 1 (1.21E-8) of magnitude time_series is\
                defined with too many digits; it must be an integer multiple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 22.1e-9],
            "time point 1 (1.21E-8) of magnitude time_series is\
                defined with too many digits; it must be an integer multiple of 1E-9",
        ),
        (
            [0.0, 22.1e-9, 12.1e-9],
            "time point 1 (2.21E-8) of magnitude time_series is\
                defined with too many digits; it must be an integer multiple of 1E-9",
        ),
    ],
)
# Rule: The times for any component of the effective Hamiltonian have
# a maximum precision of rydberg.global.time_resolution
def test_shifting_field_magnitude_time_precision_is_correct(
    times, error_message, mock_local_detuning_data, capabilities_with_local_rydberg
):
    mock_local_detuning_data["magnitude"]["time_series"]["times"] = times
    _assert_local_detuning(mock_local_detuning_data, error_message, capabilities_with_local_rydberg)


@pytest.mark.parametrize(
    "times, error_message",
    [
        (
            [0.0, 1.0, 2.0],
            "The values of the shifting field magnitude time series at\
                the first and last time points are 0.0, 2.0; they both must be both 0.",
        ),
        (
            [0.2, 1.0, 0],
            "The values of the shifting field magnitude time series at\
                the first and last time points are 0.2, 0; they both must be both 0.",
        ),
        (
            [0.0, 0.0, 1e-5],
            "The values of the shifting field magnitude time series at\
                the first and last time points are 0.0, 1e-05; they both must be both 0.",
        ),
    ],
)
def test_shifting_start_and_end_are_zero(
    times, error_message, mock_local_detuning_data, capabilities_with_local_rydberg
):
    mock_local_detuning_data["magnitude"]["time_series"]["values"] = times
    _assert_local_detuning(mock_local_detuning_data, error_message, capabilities_with_local_rydberg)


def _assert_local_detuning(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        DeviceLocalDetuningValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)
