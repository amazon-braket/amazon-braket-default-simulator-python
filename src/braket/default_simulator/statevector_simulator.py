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

from braket.default_simulator.simulator import BaseLocalSimulator
from braket.default_simulator.statevector_simulation import StateVectorSimulation
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.jaqcd import Program
from braket.task_result import GateModelTaskResult


class DefaultSimulator(BaseLocalSimulator):
    def run(
        self, circuit_ir: Program, qubit_count: int, shots: int = 0, batch_size: int = 1,
    ) -> GateModelTaskResult:
        """ Executes the circuit specified by the supplied `circuit_ir` on the simulator.

        Args:
            circuit_ir (Program): ir representation of a braket circuit specifying the
                instructions to execute.
            qubit_count (int): The number of qubits to simulate.
            shots (int): The number of times to run the circuit.
            batch_size (int): The size of the circuit partitions to contract,
                if applying multiple gates at a time is desired; see `StateVectorSimulation`.
                Must be a positive integer.
                Defaults to 1, which means gates are applied one at a time without any
                optmized contraction.

        Returns:
            GateModelTaskResult: object that represents the result

        Raises:
            ValueError: If result types are not specified in the IR or sample is specified
                as a result type when shots=0. Or, if statevector and amplitude result types
                are requested when shots>0.

        Examples:
            >>> circuit_ir = Circuit().h(0).to_ir()
            >>> DefaultSimulator().run(circuit_ir, qubit_count=1, shots=100)

            >>> circuit_ir = Circuit().h(0).to_ir()
            >>> DefaultSimulator().run(circuit_ir, qubit_count=1, batch_size=10)
        """

        simulation = StateVectorSimulation(qubit_count, shots, batch_size=batch_size)

        return BaseLocalSimulator.run(self, circuit_ir, qubit_count, shots, simulation)

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
