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

from braket.default_simulator.gate_operations import from_braket_instruction
from braket.default_simulator.observables import Hermitian, TensorProduct
from braket.default_simulator.operation import Observable
from braket.default_simulator.result_types import (
    ObservableResultType,
    ResultType,
    from_braket_result_type,
)
from braket.default_simulator.simulation import StateVectorSimulation
from braket.ir.jaqcd import Program
from braket.simulator import BraketSimulator


class DefaultSimulator(BraketSimulator):
    def run(
        self, circuit_ir: Program, qubit_count: int, shots: int = 0, *, batch_size: int = 1,
    ) -> Dict[str, Any]:
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
            Dict[str, Any]: dictionary containing the state vector (keyed by `StateVector`,
            value type `Dict[str, complex]`), measurements (keyed by `Measurements`,
            value type `List[List[str]]`, and task metadata, if any (keyed by `TaskMetadata`,
            value type `Dict[str, Any]`)).

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
        DefaultSimulator._validate_shots_and_ir_results(shots, circuit_ir)

        operations = [
            from_braket_instruction(instruction) for instruction in circuit_ir.instructions
        ]

        if shots > 0 and circuit_ir.basis_rotation_instructions:
            for instruction in circuit_ir.basis_rotation_instructions:
                operations.append(from_braket_instruction(instruction))

        simulation = StateVectorSimulation(qubit_count, shots, batch_size=batch_size)
        simulation.evolve(operations)

        results = []

        if not shots and circuit_ir.results:
            (
                non_observable_result_types,
                observable_result_types,
            ) = DefaultSimulator._translate_result_types(circuit_ir)
            observables = DefaultSimulator._validate_and_consolidate_observable_result_types(
                list(observable_result_types.values()), qubit_count
            )
            results = DefaultSimulator._generate_results(
                circuit_ir,
                non_observable_result_types,
                observable_result_types,
                observables,
                simulation,
            )

        return DefaultSimulator._create_results_dict(results, circuit_ir, simulation)

    @staticmethod
    def _validate_shots_and_ir_results(shots: int, circuit_ir: Program):
        if not shots:
            if not circuit_ir.results:
                raise ValueError("Result types must be specified in the IR when shots=0")
            for rt in circuit_ir.results:
                if rt.type in ["sample"]:
                    raise ValueError("sample can only be specified when shots>0")
        elif shots and circuit_ir.results:
            for rt in circuit_ir.results:
                if rt.type in ["statevector", "amplitude"]:
                    raise ValueError(
                        "statevector and amplitude result types not available when shots>0"
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
            none_observable_mapping[DefaultSimulator._observable_hash(obs)] = obs
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
            new_obs = DefaultSimulator._observable_hash(obs_obj)
            if measured_qubits is None:
                measured_qubits = list(range(qubit_count))
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
            return str(observable.matrix)
        elif isinstance(observable, TensorProduct):
            return ",".join(DefaultSimulator._observable_hash(obs) for obs in observable.factors)
        else:
            return str(observable.__class__.__name__)

    @staticmethod
    def _generate_results(
        circuit_ir: Program,
        non_observable_result_types: Dict[int, ResultType],
        observable_result_types: Dict[int, ObservableResultType],
        observables: List[Observable],
        simulation,
    ) -> List[Dict[str, Any]]:

        results = [0] * len(circuit_ir.results)

        for index in non_observable_result_types:
            results[index] = {
                "Type": vars(circuit_ir.results[index]),
                "Value": non_observable_result_types[index].calculate(simulation),
            }

        if observable_result_types:
            simulation.apply_observables(observables)
            for index in observable_result_types:
                results[index] = {
                    "Type": vars(circuit_ir.results[index]),
                    "Value": observable_result_types[index].calculate(simulation),
                }
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

    @staticmethod
    def _create_results_dict(
        results: List[Dict[str, Any]], circuit_ir: Program, simulation: StateVectorSimulation,
    ) -> Dict[str, Any]:
        return_dict = {
            "TaskMetadata": {
                "Id": str(uuid.uuid4()),
                "Ir": circuit_ir.json(),
                "IrType": "jaqcd",
                "Shots": simulation.shots,
            }
        }
        if results:
            return_dict["ResultTypes"] = results
        if simulation.shots:
            return_dict["Measurements"] = DefaultSimulator._formatted_measurements(simulation)
            return_dict["MeasuredQubits"] = DefaultSimulator._get_measured_qubits(
                simulation.qubit_count
            )

        return return_dict

    @property
    def properties(self) -> Dict[str, Any]:
        observables = ["X", "Y", "Z", "H", "I", "Hermitian"]
        max_shots = 10000000
        return {
            "supportedQuantumOperations": sorted(
                [
                    instruction.__name__
                    for instruction in from_braket_instruction.registry
                    if type(instruction) is not type
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
                {"name": "StateVector", "minShots": 0, "maxShots": 0},
                {"name": "Amplitude", "minShots": 0, "maxShots": 0},
            ],
        }
