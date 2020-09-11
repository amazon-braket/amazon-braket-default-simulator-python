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

import uuid
from typing import Any, Dict, List, Tuple

from braket.default_simulator.observables import Hermitian, TensorProduct
from braket.default_simulator.operation import Observable, Operation
from braket.default_simulator.operation_helpers import from_braket_instruction
from braket.default_simulator.result_types import (
    ObservableResultType,
    ResultType,
    from_braket_result_type,
)
from braket.default_simulator.state_vector_simulation import StateVectorSimulation
from braket.device_schema.device_action_properties import DeviceActionType
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.ir.jaqcd import Program
from braket.simulator import BraketSimulator
from braket.task_result import (
    AdditionalMetadata,
    GateModelTaskResult,
    ResultTypeValue,
    TaskMetadata,
)


class BaseLocalSimulator(BraketSimulator):
    def run(
        self, circuit_ir: Program, qubit_count: int, shots: int, *, batch_size: int = 1,
    ) -> GateModelTaskResult:
        """ Executes the circuit specified by the supplied `circuit_ir` on the simulator.

        Args:
            circuit_ir (Program): ir representation of a braket circuit specifying the
                instructions to execute.
            qubit_count (int): The number of qubits to simulate.
            shots (int): The number of times to run the circuit.
            simulation (Simulation): Simulation method for evolving the state.
            batch_size (int): The size of the circuit partitions to contract,
                if applying multiple gates at a time is desired; see `StateVectorSimulation`.
                Must be a positive integer.
                Defaults to 1, which means gates are applied one at a time without any
                optimized contraction.
        Returns:
            GateModelTaskResult: object that represents the result

        Raises:
            ValueError: If result types are not specified in the IR or sample is specified
                as a result type when shots=0. Or, if statevector and amplitude result types
                are requested when shots>0.
        """
        self._validate_ir_results_compatibility(circuit_ir)
        self._validate_ir_instructions_compatibility(circuit_ir)
        BaseLocalSimulator._validate_shots_and_ir_results(shots, circuit_ir, qubit_count)

        operations = [
            from_braket_instruction(instruction) for instruction in circuit_ir.instructions
        ]

        if shots > 0 and circuit_ir.basis_rotation_instructions:
            for instruction in circuit_ir.basis_rotation_instructions:
                operations.append(from_braket_instruction(instruction))

        BaseLocalSimulator._validate_operation_qubits(operations)

        simulation = self.initialize_simulation(
            qubit_count=qubit_count, shots=shots, batch_size=batch_size
        )
        simulation.evolve(operations)

        results = []

        if not shots and circuit_ir.results:
            (
                non_observable_result_types,
                observable_result_types,
            ) = BaseLocalSimulator._translate_result_types(circuit_ir)
            observables = BaseLocalSimulator._validate_and_consolidate_observable_result_types(
                list(observable_result_types.values()), qubit_count
            )
            results = BaseLocalSimulator._generate_results(
                circuit_ir,
                non_observable_result_types,
                observable_result_types,
                observables,
                simulation,
            )

        return self._create_results_obj(results, circuit_ir, simulation)

    def _validate_ir_results_compatibility(self, circuit_ir):
        if circuit_ir.results:
            circuit_result_types_name = [result.__class__.__name__ for result in circuit_ir.results]
            supported_result_types = self.properties.action[
                DeviceActionType.JAQCD
            ].supportedResultTypes
            supported_result_types_name = [result.name for result in supported_result_types]
            for name in circuit_result_types_name:
                if name not in supported_result_types_name:
                    raise TypeError(
                        f"result type {name} is not supported by {self.__class__.__name__}"
                    )

    def _validate_ir_instructions_compatibility(self, circuit_ir):
        circuit_instructions_name = [instr.__class__.__name__ for instr in circuit_ir.instructions]
        supported_instructions_name = self.properties.action[
            DeviceActionType.JAQCD
        ].supportedOperations
        for name in circuit_instructions_name:
            if name not in supported_instructions_name:
                raise TypeError(f"instruction {name} is not supported by {self.__class__.__name__}")

    @staticmethod
    def _validate_shots_and_ir_results(shots: int, circuit_ir: Program, qubit_count: int) -> None:
        if not shots:
            if not circuit_ir.results:
                raise ValueError("Result types must be specified in the IR when shots=0")
            for rt in circuit_ir.results:
                if rt.type in ["sample"]:
                    raise ValueError("sample can only be specified when shots>0")
                if rt.type == "amplitude":
                    BaseLocalSimulator._validate_amplitude_states(rt.states, qubit_count)
        elif shots and circuit_ir.results:
            for rt in circuit_ir.results:
                if rt.type in ["statevector", "amplitude", "densitymatrix"]:
                    raise ValueError(
                        "statevector, amplitude and densitymatrix result"
                        "types not available when shots>0"
                    )

    @staticmethod
    def _validate_amplitude_states(states: List[str], qubit_count: int):
        for state in states:
            if len(state) != qubit_count:
                raise ValueError(
                    f"Length of state {state} for result type amplitude"
                    f" must be equivalent to number of qubits {qubit_count} in circuit"
                )

    @staticmethod
    def _validate_operation_qubits(operations: List[Operation]) -> None:
        qubits_referenced = {target for operation in operations for target in operation.targets}
        if max(qubits_referenced) >= len(qubits_referenced):
            raise ValueError(
                "Non-contiguous qubit indices supplied; "
                "qubit indices in a circuit must be contiguous."
            )

    @staticmethod
    def _get_measured_qubits(qubit_count: int) -> List[int]:
        return list(range(qubit_count))

    @staticmethod
    def _translate_result_types(
        circuit_ir: Program,
    ) -> Tuple[Dict[int, ResultType], Dict[int, ObservableResultType]]:
        non_observable_result_types = {}
        observable_result_types = {}
        for i in range(len(circuit_ir.results)):
            result_type = from_braket_result_type(circuit_ir.results[i])
            if isinstance(result_type, ObservableResultType):
                observable_result_types[i] = result_type
            else:
                non_observable_result_types[i] = result_type
        return non_observable_result_types, observable_result_types

    @staticmethod
    def _validate_and_consolidate_observable_result_types(
        observable_result_types: List[ObservableResultType], qubit_count: int
    ) -> List[Observable]:
        none_observables = (
            rt.observable for rt in observable_result_types if rt.observable.measured_qubits is None
        )
        none_observable_mapping = {}
        for obs in none_observables:
            none_observable_mapping[BaseLocalSimulator._observable_hash(obs)] = obs
        unique_none_observables = list(none_observable_mapping.values())
        if len(unique_none_observables) > 1:
            raise ValueError(
                f"All qubits are already being measured in {unique_none_observables[0]};"
                f"cannot measure in {unique_none_observables[1:]}"
            )
        not_none_observable_list = []
        qubit_observable_mapping = {}
        for result_type in observable_result_types:
            obs_obj = result_type.observable
            measured_qubits = obs_obj.measured_qubits
            new_obs = BaseLocalSimulator._observable_hash(obs_obj)
            if measured_qubits is None:
                measured_qubits = list(range(qubit_count))

            if max(measured_qubits) >= qubit_count:
                raise ValueError(
                    f"Result type ({result_type.__class__.__name__}) Observable "
                    f"({obs_obj.__class__.__name__}) references invalid qubits {measured_qubits}"
                )
            duplicate = False
            for qubit in measured_qubits:
                # Validate that the same observable is requested for a qubit in the result types
                existing_obs = qubit_observable_mapping.get(qubit)
                if existing_obs:
                    duplicate = True
                    if existing_obs != new_obs:
                        raise ValueError(
                            f"Qubit {qubit} is already being measured in {existing_obs};"
                            f" cannot measure in {new_obs}."
                        )
                else:
                    qubit_observable_mapping[qubit] = new_obs
            if not duplicate and not none_observable_mapping.get(new_obs):
                not_none_observable_list.append(obs_obj)
        return not_none_observable_list + unique_none_observables

    @staticmethod
    def _observable_hash(observable: Observable) -> str:
        if isinstance(observable, Hermitian):
            return str(hash(str(observable.matrix.tostring())))
        elif isinstance(observable, TensorProduct):
            return ",".join(BaseLocalSimulator._observable_hash(obs) for obs in observable.factors)
        else:
            return str(observable.__class__.__name__)

    @staticmethod
    def _generate_results(
        circuit_ir: Program,
        non_observable_result_types: Dict[int, ResultType],
        observable_result_types: Dict[int, ObservableResultType],
        observables: List[Observable],
        simulation,
    ) -> List[ResultTypeValue]:

        results = [0] * len(circuit_ir.results)

        for index in non_observable_result_types:
            results[index] = ResultTypeValue.construct(
                type=circuit_ir.results[index],
                value=non_observable_result_types[index].calculate(simulation),
            )

        if observable_result_types:
            simulation.apply_observables(observables)
            for index in observable_result_types:
                results[index] = ResultTypeValue.construct(
                    type=circuit_ir.results[index],
                    value=observable_result_types[index].calculate(simulation),
                )
        return results

    @staticmethod
    def _formatted_measurements(simulation: StateVectorSimulation) -> List[List[str]]:
        """ Retrieves formatted measurements obtained from the specified simulation.

        Args:
            simulation (StateVectorSimulation): Simulation to use for obtaining the measurements.

        Returns:
            List[List[str]]: List containing the measurements, where each measurement consists
            of a list of measured values of qubits.
        """
        return [
            list("{number:0{width}b}".format(number=sample, width=simulation.qubit_count))
            for sample in simulation.retrieve_samples()
        ]

    def _create_results_obj(
        self, results: List[Dict[str, Any]], circuit_ir: Program, simulation: StateVectorSimulation,
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
            result_dict["measurements"] = BaseLocalSimulator._formatted_measurements(simulation)
            result_dict["measuredQubits"] = BaseLocalSimulator._get_measured_qubits(
                simulation.qubit_count
            )

        return GateModelTaskResult.construct(**result_dict)

    @property
    def simulation_type(self):
        raise NotImplementedError("simulation_type has not been implemented yet.")

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        """ properties of simulator such as supported IR types, quantum operations,
        and result types.
        """
        raise NotImplementedError("properties has not been implemented yet.")
