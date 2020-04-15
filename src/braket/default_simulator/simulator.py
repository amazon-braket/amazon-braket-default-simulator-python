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
import itertools
from typing import Any, Dict, List, Tuple

from braket.default_simulator.gate_operations import from_braket_instruction
from braket.default_simulator.result_types import (
    ObservableResultType,
    ResultType,
    Sample,
    from_braket_result_type,
)
from braket.default_simulator.simulation import StateVectorSimulation
from braket.ir.jaqcd import Program


class DefaultSimulator:
    def run(self, circuit_ir: Program, qubit_count: int, shots: int) -> Dict[str, Any]:
        """ Executes the circuit specified by the supplied `circuit_ir` on the simulator.

        Args:
            circuit_ir (Program): ir representation of a braket circuit specifying the
                instructions to execute.
            qubit_count (int): The number of qubits to simulate.
            shots (int): The number of times to run the circuit.

        Returns:
            Dict[str, Any]: dictionary containing the state vector (keyed by `StateVector`,
            value type `Dict[str, complex]`), measurements (keyed by `Measurements`,
            value type `List[List[str]]`, and task metadata, if any (keyed by `TaskMetadata`,
            value type `Dict[str, Any]`)).

        Examples:
            >>> circuit_ir = Circuit().h(0).to_ir()
            >>> DefaultSimulator().run(circuit_ir, qubit_count=1, shots=100)
        """
        operations = [
            from_braket_instruction(instruction) for instruction in circuit_ir.instructions
        ]

        (
            non_observable_result_types,
            observable_result_types,
        ) = DefaultSimulator._translate_result_types(circuit_ir)
        DefaultSimulator._validate_observable_result_types(
            list(observable_result_types.values()), shots
        )

        simulation = StateVectorSimulation(qubit_count, shots)
        simulation.evolve(operations)

        results = [
            {
                "Type": vars(circuit_ir.results[index]),
                "Value": non_observable_result_types[index].calculate(simulation),
            }
            for index in non_observable_result_types
        ]

        if observable_result_types:
            observables = [
                observable_result_types[index].observable for index in observable_result_types
            ]
            simulation.apply_observables(observables)
            results += [
                {
                    "Type": vars(circuit_ir.results[index]),
                    "Value": observable_result_types[index].calculate(simulation),
                }
                for index in observable_result_types
            ]

        return_dict = {"TaskMetadata": None}
        if results:
            return_dict["ResultTypes"] = results
        if shots:
            return_dict["Measurements"] = DefaultSimulator._formatted_measurements(simulation)

        return return_dict

    @staticmethod
    def _translate_result_types(
        circuit_ir: Program,
    ) -> Tuple[Dict[int, ResultType], Dict[int, ObservableResultType]]:
        if not circuit_ir.results:
            return {}, {}
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
        if [
            result_type
            for result_type in observable_result_types
            if isinstance(result_type, Sample)
        ] and not shots:
            raise ValueError("No shots specified for sample measurement")

        # Validate that if no target is specified for an observable
        # (and so the observable acts on all qubits), then it is the
        # only observable.
        observable_targets = [
            result_type.observable.targets for result_type in observable_result_types
        ]
        if None in observable_targets and len(observable_result_types) > 1:
            raise ValueError(
                "Only one observable is allowed when one acts on all targets, but "
                f"{len(observable_result_types)} observables were found"
            )

        # Validate that there are no overlapping observable targets
        flattened = list(itertools.chain(*observable_targets))
        if len(flattened) != len(set(flattened)):
            raise ValueError(
                "Overlapping targets among observables; qubits with more than one observable: "
                f"{set([qubit for qubit in flattened if flattened.count(qubit) > 1])}"
            )

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
