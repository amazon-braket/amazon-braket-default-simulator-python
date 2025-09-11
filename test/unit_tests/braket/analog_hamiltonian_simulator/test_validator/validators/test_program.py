# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import pytest
from braket.ir.ahs.program_v1 import Program
from pydantic.v1.error_wrappers import ValidationError

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
            "localDetuning": [
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


# False example with net detuning larger than the MAX_NET_DETUNING
@pytest.fixture
def mock_program_with_large_net_detuning_data():
    data = {
        "setup": {
            "ahs_register": {
                "sites": [[0, 0], [0, 1e-6]],
                "filling": [1, 1],
            }
        },
        "hamiltonian": {
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
                        "time_series": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [-125000000, -125000000, 125000000, 125000000],
                        },
                        "pattern": [0.0, 1.0],
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


def test_program_local_detuning_pattern_has_the_same_length_as_atom_array_sites(
    mock_program_data: Program, device_capabilities_constants
):
    mock_program_data.setup.ahs_register.sites = [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]]
    mock_program_data.hamiltonian.localDetuning = [
        {
            "magnitude": {
                "time_series": {"times": [], "values": []},
                "pattern": [0.0, 1.0, 0.5],
            }
        }
    ]
    error_message = "The length of pattern (3) of local detuning 0 must equal the number "
    "of atom array sites (4)."
    _assert_program(mock_program_data.dict(), error_message, device_capabilities_constants)


def test_mock_program_with_large_net_detuning_data(
    mock_program_with_large_net_detuning_data: Program, device_capabilities_constants
):
    warning_message = (
        f"Atom {1} has net detuning {-250000000.0} rad/s "
        f"at time {0} seconds, which is outside the typical range "
        f"[{-device_capabilities_constants.MAX_NET_DETUNING}, "
        f"{device_capabilities_constants.MAX_NET_DETUNING}]."
        f"Numerical instabilities may occur during simulation."
    )

    with pytest.warns(UserWarning) as e:
        ProgramValidator(
            capabilities=device_capabilities_constants,
            **mock_program_with_large_net_detuning_data.dict(),
        )
    assert warning_message in str(e[-1].message)


def _assert_program(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        ProgramValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)
