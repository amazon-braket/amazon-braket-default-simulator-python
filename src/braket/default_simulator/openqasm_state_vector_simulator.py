import sys
import uuid

from braket.device_schema import (
    DeviceActionType,
    DeviceCapabilities,
    DeviceServiceProperties,
    OpenQASMDeviceActionProperties,
)
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorParadigmProperties,
)
from braket.ir.openqasm import Program
from braket.task_result import AdditionalMetadata, TaskMetadata
from braket.task_result.oq3_program_result_v1 import OQ3ProgramResult

from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.simulator import BaseLocalSimulator


class OpenQASMStateVectorSimulator(BaseLocalSimulator):
    DEVICE_ID = "braket_oq3_native_sv"

    @property
    def device_action_type(self):
        return DeviceActionType.OPENQASM

    # def initialize_simulation(self):
    #     # any preprocessing passes could go here
    #     return OpenQASMStateVectorSimulation()

    def run(
        self,
        openqasm_ir: Program,
        shots: int = 0,
    ) -> OQ3ProgramResult:
        if openqasm_ir.source.endswith(".qasm"):
            context = Interpreter().run_file(
                openqasm_ir.source,
                shots=shots,
                inputs=openqasm_ir.inputs,
            )
        else:
            context = Interpreter().run(
                openqasm_ir.source,
                shots=shots,
                inputs=openqasm_ir.inputs,
            )
        self._validate_ir_results_compatibility(context.results)
        self._validate_shots_and_ir_results(shots, context.results, context.num_qubits)

        results = []
        if not shots and context.results:
            result_types = BaseLocalSimulator._translate_result_types(context.results)
            results = BaseLocalSimulator._generate_results(
                context.results,
                result_types,
                context.quantum_simulation,
            )

        return self._create_results_obj(results, openqasm_ir, shots, context)

    @property
    def properties(self) -> DeviceCapabilities:
        max_shots = sys.maxsize
        observables = ["x", "y", "z", "h", "i", "hermitian"]
        return GateModelSimulatorDeviceCapabilities(
            action={
                DeviceActionType.OPENQASM: OpenQASMDeviceActionProperties(
                    actionType="braket.ir.openqasm.program",
                    supportedOperations=[],
                    supportedResultTypes=[
                        {"name": "StateVector", "minShots": 0, "maxShots": 0},
                        {"name": "Amplitude", "minShots": 0, "maxShots": 0},
                        {"name": "DensityMatrix", "minShots": 0, "maxShots": 0},
                        {"name": "Probability", "minShots": 0, "maxShots": max_shots},
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
                    ],
                    version=["1"],
                ),
            },
            service=DeviceServiceProperties(
                executionWindows=[],
                shotsRange=(0, 50),
            ),
            deviceParameters={},
            paradigm=GateModelSimulatorParadigmProperties(
                qubitCount=50,
            ),
        )

    def _create_results_obj(
        self,
        results,
        openqasm_ir,
        shots,
        context,
    ) -> OQ3ProgramResult:
        return OQ3ProgramResult(
            taskMetadata=TaskMetadata(
                id=str(uuid.uuid4()),
                shots=shots,
                deviceId=self.DEVICE_ID,
            ),
            additionalMetadata=AdditionalMetadata(
                action=openqasm_ir,
            ),
            outputVariables={key: list(vals) for key, vals in context.shot_data.items()},
            resultTypes=results,
        )

    def initialize_simulation(self, **kwargs):
        pass
