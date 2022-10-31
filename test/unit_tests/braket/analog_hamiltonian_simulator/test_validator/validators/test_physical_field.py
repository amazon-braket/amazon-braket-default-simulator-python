import pytest
from pydantic.error_wrappers import ValidationError

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
