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

import sys

from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)

from braket.default_simulator import StateVectorSimulation
from braket.default_simulator.openqasm_native_simulator import BaseLocalOQ3NativeSimulator


class OpenQASMNativeStateVectorSimulator(BaseLocalOQ3NativeSimulator):
    DEVICE_ID = "braket_oq3_native_sv"

    def initialize_simulation(self, **kwargs) -> StateVectorSimulation:
        """
        Initialize state vector simulation.

        Kwargs:
            qubit_count (int), shots (int), batch_size (int)

        Returns:
            StateVectorSimulation: Initialized simulation.
        """
        qubit_count = kwargs.get("qubit_count")
        shots = kwargs.get("shots")
        batch_size = kwargs.get("batch_size")
        return StateVectorSimulation(qubit_count, shots, batch_size)

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        """
        Device properties for the OpenQASMDensityMatrixSimulator.

        Returns:
            GateModelSimulatorDeviceCapabilities: Device capabilities for this simulator.
        """
        observables = ["x", "y", "z", "h", "i", "hermitian"]
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
                    "braket.ir.openqasm.program": {
                        "actionType": "braket.ir.openqasm.program",
                        "version": ["1"],
                        "supportedOperations": sorted(
                            [
                                # OpenQASM primitives
                                "U",
                                "GPhase",
                                # builtin Braket gates
                                "ccnot",
                                "cnot",
                                "cphaseshift",
                                "cphaseshift00",
                                "cphaseshift01",
                                "cphaseshift10",
                                "cswap",
                                "cv",
                                "cy",
                                "cz",
                                "ecr",
                                "h",
                                "i",
                                "iswap",
                                "pswap",
                                "phaseshift",
                                "rx",
                                "ry",
                                "rz",
                                "s",
                                "si",
                                "swap",
                                "t",
                                "ti",
                                "unitary",
                                "v",
                                "vi",
                                "x",
                                "xx",
                                "xy",
                                "y",
                                "yy",
                                "z",
                                "zz",
                                # noise operations
                                "bit_flip",
                                "phase_flip",
                                "pauli_channel",
                                "depolarizing",
                                "two_qubit_depolarizing",
                                "two_qubit_dephasing",
                                "amplitude_damping",
                                "generalized_amplitude_damping",
                                "phase_damping",
                                "kraus",
                            ]
                        ),
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
