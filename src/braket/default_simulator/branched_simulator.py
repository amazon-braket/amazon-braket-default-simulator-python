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

import sys

from braket.default_simulator.simulator import BaseLocalSimulator
from braket.default_simulator.branched_simulation import BranchedSimulation
from braket.default_simulator.openqasm.branched_interpreter import BranchedInterpreter
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.task_result import GateModelTaskResult
from braket.device_schema import DeviceActionType


class BranchedSimulator(BaseLocalSimulator):
    DEVICE_ID = "braket_sv_branched"

    def initialize_simulation(self, **kwargs) -> BranchedSimulation:
        """
        Initialize branched simulation for mid-circuit measurements.

        Args:
            `**kwargs`: qubit_count, shots, batch_size

        Returns:
            BranchedSimulation: Initialized branched simulation.
        """
        qubit_count = kwargs.get("qubit_count", 1)
        shots = kwargs.get("shots", 1)
        batch_size = kwargs.get("batch_size", 1)
        
        if shots is None or shots <= 0:
            raise ValueError("Branched simulator requires shots > 0 for mid-circuit measurements")
            
        return BranchedSimulation(qubit_count, shots, batch_size)

    def create_program_context(self):
        """Override to prevent standard AST traversal"""
        # Return None to indicate we'll handle AST traversal ourselves
        return None

    def parse_program(self, program: OpenQASMProgram):
        """Override to skip standard parsing - we'll handle AST traversal in run_openqasm"""
        # Just parse the AST structure without executing instructions
        from braket.default_simulator.openqasm.parser.openqasm_parser import parse
        
        is_file = program.source.endswith(".qasm")
        if is_file:
            with open(program.source, encoding="utf-8") as f:
                source = f.read()
        else:
            source = program.source
            
        # Parse AST but don't execute - return the parsed AST
        return parse(source)

    def run_openqasm(
        self,
        openqasm_ir: OpenQASMProgram,
        shots: int = 0,
        *,
        batch_size: int = 1,
    ) -> GateModelTaskResult:
        """
        Executes the circuit with branching simulation for mid-circuit measurements.
        
        This method overrides the base implementation to use custom AST traversal
        that handles branching at measurement points.
        """
        if shots <= 0:
            raise ValueError("Branched simulator requires shots > 0")

        # Parse the AST structure
        ast = self.parse_program(openqasm_ir)
        
        # Create branched interpreter
        interpreter = BranchedInterpreter()
        
        # Initialize simulation - we'll determine qubit count during AST traversal
        simulation = self.initialize_simulation(
            qubit_count=0,  # Will be updated during traversal
            shots=shots, 
            batch_size=batch_size
        )
        
        # Execute with branching logic
        results = interpreter.execute_with_branching(
            ast, 
            simulation, 
            openqasm_ir.inputs or {}
        )
        
        # Create result object
        return self._create_results_obj(
            results.get("result_types", []),
            openqasm_ir,
            results.get("simulation", []),
            results.get("measured_qubits", []),
            results.get("mapped_measured_qubits", [])
        )

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        """
        Device properties for the BranchedSimulator.
        Similar to StateVectorSimulator but with mid-circuit measurement support.
        """
        observables = ["x", "y", "z", "h", "i", "hermitian"]
        max_shots = sys.maxsize
        qubit_count = 26
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
                    "shotsRange": [1, max_shots],  # Require at least 1 shot
                },
                "action": {
                    "braket.ir.openqasm.program": {
                        "actionType": "braket.ir.openqasm.program",
                        "version": ["1"],
                        "supportedOperations": [
                            # OpenQASM primitives
                            "U",
                            "GPhase",
                            # builtin Braket gates
                            "ccnot",
                            "cnot",
                            "cphaseshift",
                            "cphaseshift00",
                            "cphaseshift01",
                            "cphaseshift10",
                            "cswap",
                            "cv",
                            "cy",
                            "cz",
                            "ecr",
                            "gpi",
                            "gpi2",
                            "h",
                            "i",
                            "iswap",
                            "ms",
                            "pswap",
                            "phaseshift",
                            "prx",
                            "rx",
                            "ry",
                            "rz",
                            "s",
                            "si",
                            "swap",
                            "t",
                            "ti",
                            "unitary",
                            "v",
                            "vi",
                            "x",
                            "xx",
                            "xy",
                            "y",
                            "yy",
                            "z",
                            "zz",
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
                        ],
                        "forbiddenPragmas": [
                            "braket_noise_amplitude_damping",
                            "braket_noise_bit_flip",
                            "braket_noise_depolarizing",
                            "braket_noise_kraus",
                            "braket_noise_pauli_channel",
                            "braket_noise_generalized_amplitude_damping",
                            "braket_noise_phase_flip",
                            "braket_noise_phase_damping",
                            "braket_noise_two_qubit_dephasing",
                            "braket_noise_two_qubit_depolarizing",
                            "braket_result_type_adjoint_gradient",
                        ],
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
                                "minShots": 1,
                                "maxShots": max_shots,
                            },
                            {
                                "name": "Variance",
                                "observables": observables,
                                "minShots": 1,
                                "maxShots": max_shots,
                            },
                            {"name": "Probability", "minShots": 1, "maxShots": max_shots},
                        ],
                        "supportPhysicalQubits": False,
                        "supportsPartialVerbatimBox": False,
                        "requiresContiguousQubitIndices": False,
                        "requiresAllQubitsMeasurement": False,
                        "supportsUnassignedMeasurements": True,
                        "disabledQubitRewiringSupported": False,
                        "supportsMidCircuitMeasurement": True,  # Key difference
                    },
                },
                "paradigm": {"qubitCount": qubit_count},
                "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
            }
        )
