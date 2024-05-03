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

import numpy as np
from braket.device_schema import DeviceCapabilities
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata
from pydantic.v1 import create_model  # This is temporary for defining properties below

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
    capabilities_constants,
)
from braket.analog_hamiltonian_simulator.rydberg.numpy_solver import rk_run
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    get_blockade_configurations,
    sample_state,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_result_converter import (
    convert_result,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)
from braket.analog_hamiltonian_simulator.rydberg.scipy_solver import scipy_integrate_ode_run
from braket.analog_hamiltonian_simulator.rydberg.validators.blockade_radius import (
    validate_blockade_radius,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.ir_validator import validate_program
from braket.analog_hamiltonian_simulator.rydberg.validators.rydberg_coefficient import (
    validate_rydberg_interaction_coef,
)
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.simulator import BaseLocalSimulator

from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import Performance

from braket.aws import AwsDevice 
import multiprocessing as mp

qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
capabilities = qpu.properties.paradigm

performance = capabilities.performance

from braket.analog_hamiltonian_simulator.rydberg.noise_simulation import (get_shot_measurement, convert_ir_program_back)

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from braket.aws import AwsDevice 

def ahs_noise_simulation_v2(
    program: AnalogHamiltonianSimulation,
    noise_model: Performance,
    shots: int = 100,
    steps: int = 100,
):
    task_metadata = TaskMetadata(
        id="rydberg",
        shots=shots,
        deviceId="NoisyRydbergLocalSimulator",
    )            
    
    with mp.Pool(processes=mp.cpu_count(), initializer=np.random.seed) as p:
        measurements = p.map(get_shot_measurement, [[program, noise_model, steps] for _ in range(shots)])
    
    return AnalogHamiltonianSimulationTaskResult(
        taskMetadata=task_metadata, measurements=measurements
    )


class NoisyRydbergAtomSimulator(BaseLocalSimulator):
    DEVICE_ID = "braket_ahs_noisy"

    def __init__(self, performance=None):
        super(NoisyRydbergAtomSimulator, self).__init__()
        if performance is None:
            self._qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
            self._noise_model = self._qpu.properties.paradigm.performance    
        else:
            self._noise_model = performance

    def run(
        self,
        program: Program,
        shots: int = 100,
        steps: int = 100,
        *args,
        **kwargs        
    ) -> AnalogHamiltonianSimulationTaskResult:
        # reconstruct the AHSprogram
        program_non_ir = convert_ir_program_back(program)

        return ahs_noise_simulation_v2(program_non_ir, self._noise_model, shots, steps)

    @property
    def properties(self) -> DeviceCapabilities:
        """
        Device properties for the RydbergAtomSimulator.

        Returns:
            DeviceCapabilities: Device capabilities for this simulator.
        """
        properties = {
            "service": {
                "executionWindows": [
                    {
                        "executionDay": "Everyday",
                        "windowStartHour": "00:00",
                        "windowEndHour": "23:59:59",
                    }
                ],
                "shotsRange": [0, sys.maxsize],
            },
            "action": {"braket.ir.ahs.program": {}},
        }

        RydbergSimulatorDeviceCapabilities = create_model(
            "RydbergSimulatorDeviceCapabilities", **properties
        )

        return RydbergSimulatorDeviceCapabilities.parse_obj(properties)

    def initialize_simulation(self, **kwargs) -> Simulation:
        """
        Initialize Rydberg Hamiltonian simulation.

        Returns:
            Simulation: Initialized simulation.
        """
        pass
