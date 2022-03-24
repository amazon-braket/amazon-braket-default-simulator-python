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
from braket.simulator import BraketSimulator


class OpenQASMStateVectorSimulator(BraketSimulator):
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
        return self._create_results_obj(openqasm_ir, shots, context)

    @property
    def properties(self) -> DeviceCapabilities:
        return GateModelSimulatorDeviceCapabilities(
            action={
                DeviceActionType.OPENQASM: OpenQASMDeviceActionProperties(
                    actionType="braket.ir.openqasm.program",
                    supportedOperations=[],
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
        openqasm_ir,
        shots,
        context,
    ) -> OQ3ProgramResult:
        return OQ3ProgramResult(
            taskMetadata=TaskMetadata(
                id="task-id-here",
                shots=shots,
                deviceId="braket_oq3_sv",
            ),
            additionalMetadata=AdditionalMetadata(
                action=openqasm_ir,
            ),
            outputVariables={key: list(vals) for key, vals in context.shot_data.items()},
        )
