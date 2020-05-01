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
from braket.default_simulator.result_types import (
    ObservableResultType,
    ResultType,
    from_braket_result_type,
)
from braket.default_simulator.simulation import StateVectorSimulation
from braket.ir.jaqcd import Program


class DefaultSimulator:
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
            DefaultSimulator._validate_observable_result_types(
                list(observable_result_types.values()), shots
            )
            results = DefaultSimulator._generate_results(
                circuit_ir, non_observable_result_types, observable_result_types, simulation
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
    def _validate_observable_result_types(
        observable_result_types: List[ObservableResultType], shots: int
    ) -> None:
        # Validate that if no target is specified for an observable
        # (and so the observable acts on all qubits), then it is the
        # only observable.
        flattened = []
        for result_type in observable_result_types:
            if result_type.observable.targets is None:
                if len(observable_result_types) > 1:
                    raise ValueError(
                        "Only one observable is allowed when one acts on all targets, but "
                        f"{len(observable_result_types)} observables were found"
                    )
                else:
                    flattened.append(None)
            else:
                flattened.extend(result_type.observable.targets)

        # Validate that there are no overlapping observable targets
        if len(flattened) != len(set(flattened)):
            raise ValueError(
                "Overlapping targets among observables; qubits with more than one observable: "
                f"{set([qubit for qubit in flattened if flattened.count(qubit) > 1])}"
            )

    @staticmethod
    def _generate_results(
        circuit_ir: Program,
        non_observable_result_types: Dict[int, ResultType],
        observable_result_types: Dict[int, ObservableResultType],
        simulation,
    ) -> List[Dict[str, Any]]:

        results = [0] * len(circuit_ir.results)

        for index in non_observable_result_types:
            results[index] = {
                "Type": vars(circuit_ir.results[index]),
                "Value": non_observable_result_types[index].calculate(simulation),
            }

        if observable_result_types:
            observables = [
                observable_result_types[index].observable for index in observable_result_types
            ]
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
