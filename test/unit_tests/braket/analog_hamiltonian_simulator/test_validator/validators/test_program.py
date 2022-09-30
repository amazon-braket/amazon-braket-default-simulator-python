import pytest
from braket.ir.ahs.program_v1 import Program
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.program import ProgramValidator


@pytest.fixture
def mock_program_data():
    data = {
        "setup": {
            "ahs_register": {
                "sites": [],
                "filling": [],
            }
        },
        "hamiltonian": {
            "drivingFields": [
                {
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
            ],
            "shiftingFields": [
                {
                    "magnitude": {
                        "time_series": {
                            "times": [],
                            "values": [],
                        },
                        "pattern": [],
                    }
                }
            ],
        },
    }
    return Program.parse_obj(data)


def test_program(program_data, device_capabilities_constants):
    try:
        ProgramValidator(capabilities=device_capabilities_constants, **program_data.dict())
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


def test_program_shifting_field_pattern_has_the_same_length_as_atom_array_sites(
    mock_program_data: Program, device_capabilities_constants
):

    mock_program_data.setup.ahs_register.sites = [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]]
    mock_program_data.hamiltonian.shiftingFields = [
        {
            "magnitude": {
                "time_series": {"times": [], "values": []},
                "pattern": [0.0, 1.0, 0.5],
            }
        }
    ]
    error_message = "The length of pattern (3) of shifting field 0 must equal the number "
    "of atom array sites (4)."
    _assert_program(mock_program_data.dict(), error_message, device_capabilities_constants)


def _assert_program(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        ProgramValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)
