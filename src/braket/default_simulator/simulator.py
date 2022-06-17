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

import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Union

from braket.device_schema import DeviceCapabilities
from braket.ir.annealing import Problem
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.jaqcd.program_v1 import Results
from braket.ir.openqasm import Program as OQ3Program
from braket.task_result import AnnealingTaskResult, GateModelTaskResult, ResultTypeValue
from braket.task_result.oq3_program_result_v1 import OQ3ProgramResult

from braket.default_simulator.observables import Hermitian, TensorProduct
from braket.default_simulator.operation import Observable, Operation
from braket.default_simulator.result_types import (
    ResultType,
    TargetedResultType,
    from_braket_result_type,
)
from braket.default_simulator.simulation import Simulation
from braket.simulator import BraketSimulator

_NOISE_INSTRUCTIONS = frozenset(
    instr.lower().replace("_", "")
    for instr in [
        "amplitude_damping",
        "bit_flip",
        "depolarizing",
        "generalized_amplitude_damping",
        "kraus",
        "pauli_channel",
        "two_qubit_pauli_channel",
        "phase_flip",
        "phase_damping",
        "two_qubit_dephasing",
        "two_qubit_depolarizing",
    ]
)


class BaseLocalSimulator(BraketSimulator):
    @property
    @abstractmethod
    def device_action_type(self):
        """DeviceActionType"""

    @abstractmethod
    def run(
        self, ir: Union[JaqcdProgram, OQ3Program, Problem], *args, **kwargs
    ) -> Union[GateModelTaskResult, AnnealingTaskResult, OQ3ProgramResult]:
        """run method"""

    @property
    @abstractmethod
    def properties(self) -> DeviceCapabilities:
        """simulator properties"""

    @abstractmethod
    def initialize_simulation(self, **kwargs) -> Simulation:
        """
        Initializes simulation with keyword arguments
        """

    def _validate_ir_results_compatibility(self, results: List[Results]):
        if results:
            circuit_result_types_name = [result.__class__.__name__ for result in results]
            supported_result_types = self.properties.action[
                self.device_action_type
            ].supportedResultTypes
            supported_result_types_name = [result.name for result in supported_result_types]
            for name in circuit_result_types_name:
                if name not in supported_result_types_name:
                    raise TypeError(
                        f"result type {name} is not supported by {self.__class__.__name__}"
                    )

    @staticmethod
    def _validate_shots_and_ir_results(shots: int, results: List[Results], qubit_count: int) -> None:
        if not shots:
            if not results:
                raise ValueError("Result types must be specified in the IR when shots=0")
            for rt in results:
                if rt.type in ["sample"]:
                    raise ValueError("sample can only be specified when shots>0")
                if rt.type == "amplitude":
                    BaseLocalSimulator._validate_amplitude_states(rt.states, qubit_count)
        elif shots and results:
            for rt in results:
                if rt.type in ["statevector", "amplitude", "densitymatrix"]:
                    raise ValueError(
                        "statevector, amplitude and densitymatrix result "
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
    def _translate_result_types(results) -> List[ResultType]:
        return [from_braket_result_type(result) for result in results]

    @staticmethod
    def _generate_results(
        results,
        result_types: List[ResultType],
        simulation,
    ) -> List[ResultTypeValue]:
        return [
            ResultTypeValue.construct(
                type=results[index],
                value=result_types[index].calculate(simulation),
            )
            for index in range(len(results))
        ]

    @staticmethod
    def _validate_operation_qubits(operations: List[Operation]) -> None:
        qubits_referenced = {target for operation in operations for target in operation.targets}
        if max(qubits_referenced) >= len(qubits_referenced):
            raise ValueError(
                "Non-contiguous qubit indices supplied; "
                "qubit indices in a circuit must be contiguous."
            )

    @staticmethod
    def _validate_result_types_qubits_exist(
        targeted_result_types: List[TargetedResultType], qubit_count: int
    ):
        for result_type in targeted_result_types:
            targets = result_type.targets
            if targets and max(targets) >= qubit_count:
                raise ValueError(
                    f"Result type ({result_type.__class__.__name__})"
                    f" references invalid qubits {targets}"
                )

    def _validate_ir_instructions_compatibility(self, circuit_ir):
        circuit_instruction_names = [
            instr.__class__.__name__.lower().replace("_", "") for instr in circuit_ir.instructions
        ]
        supported_instructions = frozenset(
            op.lower().replace("_", "")
            for op in self.properties.action[self.device_action_type].supportedOperations
        )
        no_noise = True
        for name in circuit_instruction_names:
            if name in _NOISE_INSTRUCTIONS:
                no_noise = False
                if name not in supported_instructions:
                    raise TypeError(
                        "Noise instructions are not supported by the state vector simulator "
                        "(by default). You need to use the density matrix simulator: "
                        'LocalSimulator("braket_dm").'
                    )
        if no_noise and _NOISE_INSTRUCTIONS.intersection(supported_instructions):
            warnings.warn(
                "You are running a noise-free circuit on the density matrix simulator. "
                "Consider running this circuit on the state vector simulator: "
                'LocalSimulator("default") for a better user experience.'
            )

    @staticmethod
    def _get_measured_qubits(qubit_count: int) -> List[int]:
        return list(range(qubit_count))

    @staticmethod
    def _tensor_product_index_dict(
        observable: TensorProduct, callable: Callable[[Observable], Any]
    ) -> Dict[int, Any]:
        obj_dict = {}
        i = 0
        factors = list(observable.factors)
        total = len(factors[0].measured_qubits)
        while factors:
            if i >= total:
                factors.pop(0)
                if factors:
                    total += len(factors[0].measured_qubits)
            if factors:
                obj_dict[i] = callable(factors[0])
            i += 1
        return obj_dict

    @staticmethod
    def _observable_hash(observable: Observable) -> Union[str, Dict[int, str]]:
        if isinstance(observable, Hermitian):
            return str(hash(str(observable.matrix.tostring())))
        elif isinstance(observable, TensorProduct):
            # Dict of target index to observable hash
            return BaseLocalSimulator._tensor_product_index_dict(
                observable, BaseLocalSimulator._observable_hash
            )
        else:
            return str(observable.__class__.__name__)

    @staticmethod
    def _formatted_measurements(simulation: Simulation) -> List[List[str]]:
        """Retrieves formatted measurements obtained from the specified simulation.

        Args:
            simulation (Simulation): Simulation to use for obtaining the measurements.

        Returns:
            List[List[str]]: List containing the measurements, where each measurement consists
            of a list of measured values of qubits.
        """
        return [
            list("{number:0{width}b}".format(number=sample, width=simulation.qubit_count))
            for sample in simulation.retrieve_samples()
        ]
