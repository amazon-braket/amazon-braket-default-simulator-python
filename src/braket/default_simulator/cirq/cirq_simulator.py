import sys
import uuid

import cirq
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.task_result import GateModelTaskResult, TaskMetadata, AdditionalMetadata
from cirq import Simulator

from braket.default_simulator.cirq.cirq_program_context import CirqProgramContext
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.simulator import BraketSimulator


class CirqSimulator(BraketSimulator):
    DEVICE_ID = "cirq"

    def initialize_simulation(self):
        """
        Initialize cirq simulation.


        Returns:
            cirq.Simulator: Initialized simulation.
        """
        return Simulator()

    def run(self, circuit_ir: OpenQASMProgram, *args, **kwargs):
        return self.run_openqasm(circuit_ir, *args, **kwargs)

    def run_openqasm(
        self,
        circuit_ir: OpenQASMProgram,
        shots: int = 1,
    ) -> GateModelTaskResult:
        is_file = circuit_ir.source.endswith(".qasm")
        interpreter = Interpreter(context=CirqProgramContext())
        circuit = interpreter.build_circuit(
            source=circuit_ir.source,
            inputs=circuit_ir.inputs,
            is_file=is_file,
        )
        circuit.append(cirq.measure(*circuit.all_qubits()))
        simulation = self.initialize_simulation()
        result = simulation.run(circuit, repetitions=shots)
        return self._create_results_obj(circuit_ir, result)

    def _get_measured_qubits(self, result):
        measured_qubits = set()
        for key in result.measurements.keys():
            measured_qubits.update([qubit[qubit.index("(") + 1 : -1] for qubit in key.split(",")])
        return list(measured_qubits)

    def _create_results_obj(
        self,
        circuit_ir: OpenQASMProgram,
        result: cirq.Result,
    ) -> GateModelTaskResult:
        return GateModelTaskResult.construct(
            taskMetadata=TaskMetadata(
                id=str(uuid.uuid4()),
                shots=result.repetitions,
                deviceId=self.DEVICE_ID,
            ),
            additionalMetadata=AdditionalMetadata(
                action=circuit_ir,
            ),
            measurements=list(list(result.measurements.values())[0]),
            measuredQubits=self._get_measured_qubits(result),
        )

    @property
    def properties(self):
        max_shots = sys.maxsize
        return GateModelSimulatorDeviceCapabilities.parse_obj(
            {
                "service": {
                    "executionWindows": [
                        {
                            "executionDay": "Everyday",
                            "windowStartHour": "00:00",
                            "windowEndHour": "23:59:59",
                        }
                    ],
                    "shotsRange": [0, max_shots],
                },
                "action": {
                    "braket.ir.openqasm.program": {
                        "actionType": "braket.ir.openqasm.program",
                        "version": ["1"],
                        "supportedOperations": [
                            # builtin Braket gates
                            "ccnot",
                            "cnot",
                            "cphaseshift",
                            "cswap",
                            "h",
                            "i",
                            "iswap",
                            "rx",
                            "ry",
                            "rz",
                            "s",
                            "swap",
                            "t",
                            "x",
                            "y",
                            "z",
                        ],
                        "supportedModifiers": [
                            {
                                "name": "ctrl",
                            },
                            {
                                "name": "negctrl",
                            },
                            {
                                "name": "pow",
                                "exponent_types": ["int", "float"],
                            },
                            {
                                "name": "inv",
                            },
                        ],
                        "supportedPragmas": [
                            "braket_unitary_matrix",
                            "braket_result_type_state_vector",
                            "braket_result_type_density_matrix",
                            "braket_result_type_sample",
                            "braket_result_type_expectation",
                            "braket_result_type_variance",
                            "braket_result_type_probability",
                            "braket_result_type_amplitude",
                            "braket_noise_bit_flip",
                            "braket_noise_depolarizing",
                            "braket_noise_generalized_amplitude_damping",
                            "braket_noise_phase_flip",
                            "braket_noise_phase_damping",
                        ],
                        "forbiddenPragmas": [
                            "braket_noise_amplitude_damping",
                            "braket_noise_kraus",
                            "braket_noise_pauli_channel",
                            "braket_noise_two_qubit_dephasing",
                            "braket_noise_two_qubit_depolarizing",
                            "braket_result_type_adjoint_gradient",
                        ],
                        "supportPhysicalQubits": False,
                        "supportsPartialVerbatimBox": False,
                        "requiresContiguousQubitIndices": True,
                        "requiresAllQubitsMeasurement": True,
                        "supportsUnassignedMeasurements": True,
                        "disabledQubitRewiringSupported": False,
                    },
                },
                "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
            }
        )
