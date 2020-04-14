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
        no_observables, with_observables = (
            DefaultSimulator._validate_result_types(circuit_ir, shots)
            if circuit_ir.results
            else ([], [])
        )

        simulation = StateVectorSimulation(qubit_count, shots)
        simulation.evolve(operations)

        results = [
            {"Type": result_type.result_info, "Value": result_type.calculate(simulation)}
            for result_type in no_observables
        ]

        if with_observables:
            observables = [result_type.observable for result_type in with_observables]
            simulation.apply_observables(observables)
            results += [
                {"Type": result_type.result_info, "Value": result_type.calculate(simulation)}
                for result_type in with_observables
            ]

        return {
            # TODO: Remove state vector from return dict
            "StateVector": DefaultSimulator._formatted_state_vector(simulation),
            "Measurements": DefaultSimulator._formatted_measurements(simulation),
            "RequestedResults": results,
            "TaskMetadata": None,
        }

    @staticmethod
    def _validate_result_types(
        circuit_ir: Program, shots: int
    ) -> Tuple[List[ResultType], List[ObservableResultType]]:
        no_observables = []
        with_observables = []
        for ir_result_type in circuit_ir.results:
            result_type = from_braket_result_type(ir_result_type)
            if isinstance(result_type, ObservableResultType):
                with_observables.append(result_type)
            else:
                no_observables.append(result_type)

        if [
            result_type for result_type in with_observables if isinstance(result_type, Sample)
        ] and not shots:
            raise ValueError("No shots specified for sample measurement")

        # Validate that if no target is specified for an observable
        # (and so the observable acts on all qubits), then it is the
        # only observable.
        observable_targets = [result_type.observable.targets for result_type in with_observables]
        if None in observable_targets and len(with_observables) > 1:
            raise ValueError(
                "Multiple observables found when one observable already acts on all qubits"
            )

        # Validate that there are no overlapping observable targets
        flattened = [qubit for target in observable_targets for qubit in target]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Overlapping targets among observables")

        return no_observables, with_observables

    @staticmethod
    def _formatted_state_vector(simulation: StateVectorSimulation,) -> Dict[str, complex]:
        """ Retrieves the formatted state vector for the specified simulation.

        Args:
            simulation (StateVectorSimulation): Simulation to be used for constructing the
                formatted state vector result.
        Returns:
            Dict[str, complex]: Dictionary with key as the state represented in the big endian
            format and the value as the probability amplitude.
        """
        return {
            "{number:0{width}b}".format(number=idx, width=simulation.qubit_count): coefficient
            for idx, coefficient in enumerate(simulation.state_vector)
        }

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
