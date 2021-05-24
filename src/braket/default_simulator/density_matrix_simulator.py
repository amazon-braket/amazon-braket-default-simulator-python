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

from braket.default_simulator.density_matrix_simulation import DensityMatrixSimulation
from braket.default_simulator.simulator import BaseLocalSimulator


class DensityMatrixSimulator(BaseLocalSimulator):

    DEVICE_ID = "braket_dm"

    def initialize_simulation(self, **kwargs):
        qubit_count = kwargs.get("qubit_count")
        shots = kwargs.get("shots")
        return DensityMatrixSimulation(qubit_count, shots)

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        observables = ["X", "Y", "Z", "H", "I", "Hermitian"]
        max_shots = sys.maxsize
        qubit_count = 13
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
                            "AmplitudeDamping",
                            "BitFlip",
                            "CCNot",
                            "CNot",
                            "CPhaseShift",
                            "CPhaseShift00",
                            "CPhaseShift01",
                            "CPhaseShift10",
                            "CSwap",
                            "CY",
                            "CZ",
                            "Depolarizing",
                            "GeneralizedAmplitudeDamping",
                            "PauliChannel",
                            "H",
                            "I",
                            "ISwap",
                            "Kraus",
                            "PSwap",
                            "PhaseShift",
                            "PhaseFlip",
                            "PhaseDamping",
                            "Rx",
                            "Ry",
                            "Rz",
                            "S",
                            "Si",
                            "Swap",
                            "T",
                            "Ti",
                            "TwoQubitDephasing",
                            "TwoQubitDepolarizing",
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
                            {"name": "DensityMatrix", "minShots": 0, "maxShots": 0},
                        ],
                    }
                },
                "paradigm": {"qubitCount": qubit_count},
                "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
            }
        )
