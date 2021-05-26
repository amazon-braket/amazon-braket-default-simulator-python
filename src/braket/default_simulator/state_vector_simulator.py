# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sys

from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)

from braket.default_simulator.simulator import BaseLocalSimulator
from braket.default_simulator.state_vector_simulation import StateVectorSimulation


class StateVectorSimulator(BaseLocalSimulator):

    DEVICE_ID = "braket_sv"

    def initialize_simulation(self, **kwargs):
        qubit_count = kwargs.get("qubit_count")
        shots = kwargs.get("shots")
        batch_size = kwargs.get("batch_size")
        return StateVectorSimulation(qubit_count, shots, batch_size)

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        observables = ["X", "Y", "Z", "H", "I", "Hermitian"]
        max_shots = sys.maxsize
        qubit_count = 26
        return GateModelSimulatorDeviceCapabilities.parse_obj(
            {
                "service": {
                    "executionWindows": [
                        {
                            "executionDay": "Everyday",
                            "windowStartHour": "00:00",
                            "windowEndHour": "23:59:59",
                        }
                    ],
                    "shotsRange": [0, max_shots],
                },
                "action": {
                    "braket.ir.jaqcd.program": {
                        "actionType": "braket.ir.jaqcd.program",
                        "version": ["1"],
                        "supportedOperations": [
                            "CCNot",
                            "CNot",
                            "CPhaseShift",
                            "CPhaseShift00",
                            "CPhaseShift01",
                            "CPhaseShift10",
                            "CSwap",
                            "CY",
                            "CZ",
                            "H",
                            "I",
                            "ISwap",
                            "PSwap",
                            "PhaseShift",
                            "Rx",
                            "Ry",
                            "Rz",
                            "S",
                            "Si",
                            "Swap",
                            "T",
                            "Ti",
                            "Unitary",
                            "V",
                            "Vi",
                            "X",
                            "XX",
                            "XY",
                            "Y",
                            "YY",
                            "Z",
                            "ZZ",
                        ],
                        "supportedResultTypes": [
                            {
                                "name": "Sample",
                                "observables": observables,
                                "minShots": 1,
                                "maxShots": max_shots,
                            },
                            {
                                "name": "Expectation",
                                "observables": observables,
                                "minShots": 0,
                                "maxShots": max_shots,
                            },
                            {
                                "name": "Variance",
                                "observables": observables,
                                "minShots": 0,
                                "maxShots": max_shots,
                            },
                            {"name": "Probability", "minShots": 0, "maxShots": max_shots},
                            {"name": "StateVector", "minShots": 0, "maxShots": 0},
                            {"name": "DensityMatrix", "minShots": 0, "maxShots": 0},
                            {"name": "Amplitude", "minShots": 0, "maxShots": 0},
                        ],
                    }
                },
                "paradigm": {"qubitCount": qubit_count},
                "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
            }
        )


DefaultSimulator = StateVectorSimulator
