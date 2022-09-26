import pytest
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.ir_validator import validate_program


def test_validate_program(program_data, device_capabilities_constants):
    try:
        validate_program(program=program_data, device_capabilities=device_capabilities_constants)
    except ValidationError as e:
        pytest.fail(f"Validate program is failing : {str(e)}")
