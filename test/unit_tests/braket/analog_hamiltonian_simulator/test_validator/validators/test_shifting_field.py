import pytest
from braket.ir.ahs.shifting_field import ShiftingField
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.shifting_field import (
    ShiftingFieldValidator,
)


@pytest.fixture
def shifting_field_data():
    return {
        "magnitude": {
            "time_series": {"times": [0, 1e-07], "values": [0, 0]},
            "pattern": [0.0, 1.0, 0.5, 0.0, 1.0],
        }
    }


def test_shifting_field(shifting_field_data, device_capabilities_constants):
    try:
        ShiftingFieldValidator(capabilities=device_capabilities_constants, **shifting_field_data)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.fixture
def mock_shifting_field_data():
    data = {
        "magnitude": {
            "pattern": [],
            "time_series": {
                "times": [],
                "values": [],
            },
        }
    }
    return ShiftingField.parse_obj(data).dict()


@pytest.mark.parametrize(
    "field_name, error_message",
    [
        ("magnitude", "Pattern of shifting field must be not be a string - test"),
    ],
)
def test_shifting_field_magnitude_pattern_is_not_uniform(
    field_name, error_message, mock_shifting_field_data, device_capabilities_constants
):
    mock_shifting_field_data[field_name]["pattern"] = "test"
    _assert_shifting_field(mock_shifting_field_data, error_message, device_capabilities_constants)


@pytest.mark.parametrize(
    "pattern, error_message",
    [
        (
            [0.0, -1.0, 0.5, 0.0, 1.0],
            "magnitude pattern value 1 is -1.0; it must be between 0.0 and 1.0 (inclusive).",
        ),
        (
            [0.0, 1.0, 0.5, 0.0, 1.5],
            "magnitude pattern value 4 is 1.5; it must be between 0.0 and 1.0 (inclusive).",
        ),
    ],
)
def test_shifting_field_magnitude_pattern_within_bounds(
    pattern, error_message, mock_shifting_field_data, device_capabilities_constants
):
    mock_shifting_field_data["magnitude"]["pattern"] = pattern
    _assert_shifting_field(mock_shifting_field_data, error_message, device_capabilities_constants)


def test_shifting_field_empty(mock_shifting_field_data, device_capabilities_constants):
    try:
        ShiftingFieldValidator(
            capabilities=device_capabilities_constants, **mock_shifting_field_data
        )
    except ValidationError as e:
        pytest.fail(f"Validate shifting field empty test is failing : {str(e)}")


@pytest.mark.parametrize(
    "values, warning_message",
    [
        (
            [0.0, -5e8, 0.5e7, 0.0],
            "Value 1 (-500000000.0) in magnitude time series outside the typical range "
            "[-125000000.0, 125000000.0]. The values should  be specified in SI units.",
        ),
        (
            [0.0, -2e8, 2.6e7, 0.0],
            "Value 1 (-200000000.0) in magnitude time series outside the typical range "
            "[-125000000.0, 125000000.0]. The values should  be specified in SI units.",
        ),
    ],
)
def test_shifting_field_magnitude_values_within_range(
    values, warning_message, mock_shifting_field_data, device_capabilities_constants
):
    mock_shifting_field_data["magnitude"]["time_series"]["values"] = values
    _assert_shifting_field_warning_message(
        mock_shifting_field_data, warning_message, device_capabilities_constants
    )


def _assert_shifting_field(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        ShiftingFieldValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)


def _assert_shifting_field_warning_message(data, warning_message, device_capabilities_constants):
    with pytest.warns(UserWarning) as e:
        ShiftingFieldValidator(capabilities=device_capabilities_constants, **data)
    assert warning_message in str(e[-1].message)
