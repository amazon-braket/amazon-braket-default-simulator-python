import pytest
from pydantic.v1.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators import (
    DeviceHamiltonianValidator,
)


@pytest.fixture
def hamiltonian_data():
    return {
        "drivingFields": [
            {
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
        ],
        "localDetuning": [
            {
                "magnitude": {
                    "time_series": {"times": [0, 4e-6], "values": [0, 0]},
                    "pattern": [0.0, 1.0, 0.5, 0.0, 1.0],
                }
            }
        ],
    }


def test_hamiltonian(hamiltonian_data):
    try:
        DeviceHamiltonianValidator(**hamiltonian_data, LOCAL_RYDBERG_CAPABILITIES=True)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize("local_rydberg_exists", [(True), (False)])
def test_hamiltonian_no_detuning(local_rydberg_exists, hamiltonian_data):
    hamiltonian_data["localDetuning"].clear()
    try:
        DeviceHamiltonianValidator(
            **hamiltonian_data, LOCAL_RYDBERG_CAPABILITIES=local_rydberg_exists
        )
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


def test_no_local_rydberg_capabilities(hamiltonian_data):
    error_message = "Local detuning cannot be specified; \
1 are given. Specifying local \
detuning is an experimental capability, use Braket Direct to request access."
    with pytest.raises(ValidationError) as e:
        DeviceHamiltonianValidator(**hamiltonian_data, LOCAL_RYDBERG_CAPABILITIES=False)
    assert error_message in str(e.value)
