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

from typing import Any, Dict, List

from braket.default_simulator.operations import from_braket_instruction
from braket.default_simulator.simulation import StateVectorSimulation
from braket.ir.jaqcd import Program


class DefaultSimulator:
    def run(self, circuit_ir: Program, qubit_count: int, shots: int) -> Dict[str, Any]:
        """Executes the circuit specified by the supplied `circuit_ir` on the simulator.

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
        simulation = StateVectorSimulation(qubit_count)
        simulation.evolve(operations)

        return {
            "StateVector": DefaultSimulator._formatted_state_vector(simulation),
            "Measurements": DefaultSimulator._formatted_measurements(simulation, shots),
            "TaskMetadata": None,
        }

    @staticmethod
    def _formatted_state_vector(simulation: StateVectorSimulation,) -> Dict[str, complex]:
        """Retrieves the formatted state vector for the specified simulation.

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
    def _formatted_measurements(simulation: StateVectorSimulation, shots: int) -> List[List[str]]:
        """Retrieves `shots` number of formatted measurements obtained from the specified simulation.

        Args:
            simulation (StateVectorSimulation): Simulation to use for obtaining the measurements.
            shots (int): Number of measurements to be performed.

        Returns:
            List[List[str]]: List containing the measurements, where each measurement consists
                of a list of measured values of qubits.
        """
        return [
            list("{number:0{width}b}".format(number=sample, width=simulation.qubit_count))
            for sample in simulation.retrieve_samples(shots)
        ]
