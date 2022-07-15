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
from abc import abstractmethod
from typing import Any, Dict, List

from braket.device_schema.device_action_properties import DeviceActionType
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.ir.openqasm import Program
from braket.task_result import AdditionalMetadata, GateModelTaskResult, TaskMetadata

from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.native_interpreter import NativeInterpreter
from braket.default_simulator.result_types import TargetedResultType
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.simulator import BaseLocalSimulator


class BaseLocalOQ3NativeSimulator(BaseLocalSimulator):
    @property
    def device_action_type(self) -> DeviceActionType:
        return DeviceActionType.OPENQASM

    def run(
        self,
        openqasm_ir: Program,
        shots: int = 1,
        *,
        batch_size: int = 1,
    ) -> GateModelTaskResult:
        """Executes the circuit specified by the supplied `circuit_ir` on the simulator.

        Args:
            openqasm_ir (Program): ir representation of a braket circuit specifying the
                instructions to execute.
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
        is_file = openqasm_ir.source.endswith(".qasm")
        simulation = self.initialize_simulation(
            qubit_count=0, shots=shots, batch_size=batch_size
        )
        interpreter = NativeInterpreter(simulation=simulation)

        # self._validate_ir_results_compatibility(circuit.results)
        # self._validate_ir_instructions_compatibility(circuit)
        # BaseLocalSimulator._validate_operation_qubits(operations)


        context = interpreter.simulate(
            source=openqasm_ir.source,
            inputs=openqasm_ir.inputs,
            is_file=is_file,
            shots=shots,
        )
        return context

    def _create_results_obj(
        self,
        results: List[Dict[str, Any]],
        openqasm_ir: Program,
        simulation: Simulation,
    ) -> GateModelTaskResult:
        return GateModelTaskResult.construct(
            taskMetadata=TaskMetadata(
                id=str(uuid.uuid4()),
                shots=simulation.shots,
                deviceId=self.DEVICE_ID,
            ),
            additionalMetadata=AdditionalMetadata(
                action=openqasm_ir,
            ),
            resultTypes=results,
            measurements=BaseLocalSimulator._formatted_measurements(simulation),
            measuredQubits=BaseLocalSimulator._get_measured_qubits(simulation.qubit_count),
        )

    @property
    @abstractmethod
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        """GateModelSimulatorDeviceCapabilities: Properties of simulator such as supported IR types,
        quantum operations, and result types.
        """
