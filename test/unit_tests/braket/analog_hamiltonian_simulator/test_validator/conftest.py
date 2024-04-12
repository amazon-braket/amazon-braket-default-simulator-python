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

from braket.analog_hamiltonian_simulator.rydberg.constants import capabilities_constants


@pytest.fixture
def device_capabilities_constants():
    return capabilities_constants()


@pytest.fixture
def program_data():
    data = {
        "setup": {
            "ahs_register": {
                "sites": [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
                "filling": [1, 0, 1, 0],
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
                        "time_series": {"times": [0, 4e-6], "values": [0, 0]},
                        "pattern": [0.0, 1.0, 0.5, 0.0],
                    }
                }
            ],
        },
    }
    return Program.parse_obj(data)
