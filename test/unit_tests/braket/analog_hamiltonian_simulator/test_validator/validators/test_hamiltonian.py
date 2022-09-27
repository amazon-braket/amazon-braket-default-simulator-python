import pytest
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.hamiltonian import HamiltonianValidator


@pytest.fixture
def hamiltonian_data():
    return {
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
                    "pattern": [0.0, 1.0, 0.5, 0.0, 1.0],
                }
            }
        ],
    }


def test_hamiltonian(hamiltonian_data):
    try:
        HamiltonianValidator(**hamiltonian_data)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.fixture
def empty_hamiltonian():
    return {
        "drivingFields": [],
        "shiftingFields": [],
    }


def test_empty_hamiltonian(empty_hamiltonian):
    try:
        HamiltonianValidator(**empty_hamiltonian)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


def test_hamiltonian_max_one_driving_field():
    hamiltonian_data = {
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
            },
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
            },
        ],
        "shiftingFields": [],
    }
    error_message = "At most one driving field should be specified; 2 are given."
    _assert_hamiltonian(error_message, hamiltonian_data)


def test_hamiltonian_max_one_shifting_field():
    hamiltonian_data = {
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
                    "pattern": [0.0, 1.0, 0.5, 0.0, 1.0],
                }
            },
            {
                "magnitude": {
                    "sequence": {"times": [0, 4e-6], "values": [0, 0]},
                    "pattern": [0.0, 1.0, 0.5, 0.0, 1.0],
                }
            },
        ],
    }
    error_message = "At most one shifting field should be specified; 2 are given."
    _assert_hamiltonian(error_message, hamiltonian_data)


def test_hamiltonian_all_sequences_in_driving_and_shifting_fields_have_the_same_last_timepoint(
    hamiltonian_data,
):
    hamiltonian_data["drivingFields"][0]["amplitude"]["sequence"]["times"] = [
        0,
        1e-07,
        3.9e-06,
        5e-06,
    ]
    error_message = "The timepoints for all the sequences are not equal."
    _assert_hamiltonian(error_message, hamiltonian_data)


def _assert_hamiltonian(error_message, hamiltonian_data):
    with pytest.raises(ValidationError) as e:
        HamiltonianValidator(**hamiltonian_data)
    assert error_message in str(e.value)