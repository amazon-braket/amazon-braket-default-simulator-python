import pytest
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import capabilities_constants


@pytest.fixture
def device_capabilities_constants():
    return capabilities_constants()


@pytest.fixture
def program_data():
    data = {
        "setup": {
            "atomArray": {
                "sites": [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
                "filling": [1, 0, 1, 0],
            }
        },
        "hamiltonian": {
            "drivingFields": [
                {
                    "amplitude": {
                        "pattern": "uniform",
                        "sequence": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [0, 12566400.0, 12566400.0, 0],
                        },
                    },
                    "phase": {
                        "pattern": "uniform",
                        "sequence": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [0, 0, -16.0832, -16.0832],
                        },
                    },
                    "detuning": {
                        "pattern": "uniform",
                        "sequence": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [-125000000, -125000000, 125000000, 125000000],
                        },
                    },
                }
            ],
            "shiftingFields": [
                {
                    "magnitude": {
                        "sequence": {"times": [0, 4e-6], "values": [0, 0]},
                        "pattern": [0.0, 1.0, 0.5, 0.0],
                    }
                }
            ],
        },
    }
    return Program.parse_obj(data)
