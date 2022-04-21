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

import uuid
from typing import Any, Dict, List

from braket.device_schema.device_action_properties import DeviceActionType
from braket.ir.jaqcd import Program
from braket.task_result import (
    AdditionalMetadata,
    GateModelTaskResult,
    TaskMetadata,
)
from braket.default_simulator.operation_helpers import from_braket_instruction
from braket.default_simulator.result_types import (
    TargetedResultType,
)
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.simulator import BaseLocalSimulator



class BaseLocalJaqcdSimulator(BaseLocalSimulator):
    @property
    def device_action_type(self):
        return DeviceActionType.JAQCD

    def run(
        self,
        circuit_ir: Program,
        qubit_count: int,
        shots: int = 0,
        *,
        batch_size: int = 1,
    ) -> GateModelTaskResult:
        """Executes the circuit specified by the supplied `circuit_ir` on the simulator.

        Args:
            circuit_ir (Program): ir representation of a braket circuit specifying the
                instructions to execute.
            qubit_count (int): The number of qubits to simulate.
            shots (int): The number of times to run the circuit.
            batch_size (int): The size of the circuit partitions to contract,
                if applying multiple gates at a time is desired; see `StateVectorSimulation`.
                Must be a positive integer.
                Defaults to 1, which means gates are applied one at a time without any
                optimized contraction.
        Returns:
            GateModelTaskResult: object that represents the result

        Raises:
            ValueError: If result types are not specified in the IR or sample is specified
                as a result type when shots=0. Or, if StateVector and Amplitude result types
                are requested when shots>0.
        """
        self._validate_ir_results_compatibility(circuit_ir.results)
        self._validate_ir_instructions_compatibility(circuit_ir)
        BaseLocalSimulator._validate_shots_and_ir_results(shots, circuit_ir.results, qubit_count)

        operations = [
            from_braket_instruction(instruction) for instruction in circuit_ir.instructions
        ]

        if shots > 0 and circuit_ir.basis_rotation_instructions:
            for instruction in circuit_ir.basis_rotation_instructions:
                operations.append(from_braket_instruction(instruction))

        BaseLocalJaqcdSimulator._validate_operation_qubits(operations)

        simulation = self.initialize_simulation(
            qubit_count=qubit_count, shots=shots, batch_size=batch_size
        )
        simulation.evolve(operations)

        results = []

        if not shots and circuit_ir.results:
            result_types = BaseLocalJaqcdSimulator._translate_result_types(circuit_ir.results)
            BaseLocalJaqcdSimulator._validate_result_types_qubits_exist(
                [
                    result_type
                    for result_type in result_types
                    if isinstance(result_type, TargetedResultType)
                ],
                qubit_count,
            )
            results = BaseLocalJaqcdSimulator._generate_results(
                circuit_ir.results,
                result_types,
                simulation,
            )

        return self._create_results_obj(results, circuit_ir, simulation)

    def _create_results_obj(
        self,
        results: List[Dict[str, Any]],
        circuit_ir: Program,
        simulation: Simulation,
    ) -> GateModelTaskResult:
        result_dict = {
            "taskMetadata": TaskMetadata(
                id=str(uuid.uuid4()), shots=simulation.shots, deviceId=self.DEVICE_ID
            ),
            "additionalMetadata": AdditionalMetadata(action=circuit_ir),
        }
        if results:
            result_dict["resultTypes"] = results
        if simulation.shots:
            result_dict["measurements"] = BaseLocalJaqcdSimulator._formatted_measurements(
                simulation
            )
            result_dict["measuredQubits"] = BaseLocalJaqcdSimulator._get_measured_qubits(
                simulation.qubit_count
            )

        return GateModelTaskResult.construct(**result_dict)
