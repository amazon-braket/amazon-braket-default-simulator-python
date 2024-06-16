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

import os
import sys
import json
from typing import Union
import numpy as np
from braket.device_schema import DeviceCapabilities
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationShotMeasurement,
    AnalogHamiltonianSimulationShotMetadata,
    AnalogHamiltonianSimulationShotResult,
    AnalogHamiltonianSimulationTaskResult,
)
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
from braket.devices.local_simulator import LocalSimulator

from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import Performance

from braket.aws import AwsDevice 
import multiprocessing as mp

from pprint import pprint as pp

import subprocess
import pandas

qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
capabilities = qpu.properties.paradigm

performance = capabilities.performance

from braket.analog_hamiltonian_simulator.rydberg.noise_simulation import (get_shot_measurement, convert_ir_program_back)

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from braket.aws import AwsDevice 

class RydbergAtomTNSimulator(BaseLocalSimulator):
    DEVICE_ID = "braket_ahs_tn"

    def run(
        self,
        program: Program,
        shots: int = 1000,
        steps: int = 100,
        blockade_radius: float = 12e-6,
        my_noise_model: Performance = qpu.properties.paradigm.performance,
        if_apply_noise: bool = False,
        max_bond_dim: int = 4,
        *args,
        **kwargs
    ) -> AnalogHamiltonianSimulationTaskResult:

        task_metadata = TaskMetadata(
            id="rydberg",
            shots=shots,
            deviceId="RydbergAtomTNSimulator",
        )                

        # Validate the input
        if isinstance(program, Program) is False:
            raise TypeError("`program` has the wrong type, it has to be a Program.")
            
        # Convert the program into json and save it
        # folder = os.path.dirname(os.path.realpath(__file__))
        folder = os.getcwd()
        uuid = np.random.randint(1000000)
        folder = f"{folder}/{uuid}"
        os.mkdir(folder)
        print(folder)
        
        json_data = json.loads(program.json())
        json_string = json.dumps(json_data, indent=4) 
        filename = f"{folder}/ahs_program.json"
        with open(filename, "w") as json_file:
            json_file.write(json_string)

        # Run with Julia
        with open(f"{folder}/tn_solver.jl", "w") as text_file:
            txt = 'using BraketAHS; run_program("' + filename + '",' + f"interaction_radius={blockade_radius}, " + f"n_tau_steps={steps}, " + f"shots={shots}, " + f"max_bond_dim={max_bond_dim}, " + 'if_compute_correlators=false, if_compute_energies=false)'
            text_file.write(txt)
        subprocess.run(['julia', f'{folder}/tn_solver.jl', filename])
        
        # Get the shot measurement
        preSequence = program.setup.ahs_register.filling
        postseqs = np.array(pandas.read_csv(f"{folder}/mps_samples.csv"), dtype=int)
        postseqs = [list(item) for item in postseqs]

        measurements = []
        for postseq in postseqs:
            shot_measurement = AnalogHamiltonianSimulationShotMeasurement(
                shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
                shotResult=AnalogHamiltonianSimulationShotResult(
                    preSequence=preSequence, postSequence=postseq
                ),
            )
            measurements.append(shot_measurement)
            
        # Delete the files
        subprocess.run(['rm', '-r', folder])


        return AnalogHamiltonianSimulationTaskResult(
            taskMetadata=task_metadata, measurements=measurements
        )

    def run_batch(
        self,
        programs: list[Program],
        shots: int = 100,
        steps: int = 100,
        my_noise_model: Performance = qpu.properties.paradigm.performance,
        if_apply_noise: bool = False,
        *args,
        **kwargs
    ) -> AnalogHamiltonianSimulationTaskResult:

        # Validate the input
        if isinstance(programs, list) is False:
            raise TypeError("`program` has the wrong type, it has to be a list of Programs.")
        else:
            for item in programs:
                if isinstance(program, Program) is False:
                    raise TypeError("`program` has the wrong type, it has to be a list of Programs.")

        def _run_internal_wrap(
            program: Program,
        ) -> AnalogHamiltonianSimulationTaskResult:
            return self.run(
                program,
                shots = shots,
                steps = steps,
                my_noise_model = my_noise_model,
                if_apply_noise = if_apply_noise,
                *args,
                **kwargs
            )

        with mp.Pool(processes=mp.cpu_count(), initializer=np.random.seed) as p:
            results = p.map(_run_internal_wrap, programs)


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
