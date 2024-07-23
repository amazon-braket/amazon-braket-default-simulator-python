import pytest
from pydantic.v1.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.device_ir_validator import (
    validate_program,
)


def test_validate_program(program_data, capabilities_with_local_rydberg):
    try:
        validate_program(program=program_data, device_capabilities=capabilities_with_local_rydberg)
    except ValidationError as e:
        pytest.fail(f"Validate program is failing : {str(e)}")
