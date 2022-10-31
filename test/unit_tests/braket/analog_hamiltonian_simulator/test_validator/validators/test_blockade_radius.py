import pytest
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.blockade_radius import (
    validate_blockade_radius,
)


@pytest.fixture
def valid_blockade_radius():
    return 4e-6


def test_blockade_radius(valid_blockade_radius):
    try:
        validate_blockade_radius(valid_blockade_radius)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "invalid_blockade_radius, error_message",
    [(-4e-6, "`blockade_radius` needs to be non-negative.")],
)
def test_invalid_blockade_radius_error_message(invalid_blockade_radius, error_message):
    with pytest.raises(ValueError) as e:
        validate_blockade_radius(invalid_blockade_radius)
    assert error_message in str(e.value)


@pytest.mark.parametrize(
    "invalid_blockade_radius, warning_message",
    [
        (
            4e-7,
            "Blockade radius 4e-07 meter is smaller than the typical value (1e-06 meter). "
            "The blockade radius should be specified in SI units.",
        )
    ],
)
def test_invalid_blockade_radius_warning_message(invalid_blockade_radius, warning_message):
    with pytest.warns(UserWarning) as e:
        validate_blockade_radius(invalid_blockade_radius)
    assert warning_message in str(e[-1].message)
