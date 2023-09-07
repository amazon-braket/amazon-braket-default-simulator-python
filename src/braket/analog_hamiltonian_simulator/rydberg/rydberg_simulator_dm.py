import sys

import numpy as np
from braket.device_schema import DeviceCapabilities
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata
from pydantic import create_model  # This is temporary for defining properties below

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
    capabilities_constants,
)
from braket.analog_hamiltonian_simulator.rydberg.density_matrix_solver import (
    dm_scipy_integrate_ode_run,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    apply_SPAM_noises,
    get_blockade_configurations,
    sample_result,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_result_converter import (
    convert_result,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.blockade_radius import (
    validate_blockade_radius,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.ir_validator import validate_program
from braket.analog_hamiltonian_simulator.rydberg.validators.rydberg_coefficient import (
    validate_rydberg_interaction_coef,
)
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.simulator import BaseLocalSimulator


class RydbergAtomSimulatorDM(BaseLocalSimulator):
    DEVICE_ID = "braket_ahs_dm"

    def run(
        self,
        program: Program,
        shots: int = 100,
        steps: int = 1000,
        rydberg_interaction_coef: float = RYDBERG_INTERACTION_COEF,
        blockade_radius: float = 0.0,
        progress_bar: bool = False,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        solver_method: str = "adams",
        order: int = 12,
        nsteps: int = 1000,
        first_step: int = 0,
        max_step: int = 0,
        min_step: int = 0,
        noises: dict = {},
        *args,
        **kwargs
    ) -> AnalogHamiltonianSimulationTaskResult:
        # Validate the given program against the capabilities
        validate_program(program, capabilities_constants())

        # Validate the Rydberg interaction coefficients and blockade radius
        self.rydberg_interaction_coef = validate_rydberg_interaction_coef(
            rydberg_interaction_coef
        ) / ((SPACE_UNIT**6) / TIME_UNIT)
        self.blockade_radius = validate_blockade_radius(blockade_radius) / SPACE_UNIT

        # Convert the units of the program from SI unit to microsecond and micrometer etc.
        program = convert_unit(program)

        # Get the duration of the program. Note that there could be 0 or 1 driving field
        # (same for the shifting field) for a given program. Hence we need to checkout
        # the duration for both fields
        self.duration = 0
        if len(program.hamiltonian.drivingFields) == 1:
            self.duration = float(
                program.hamiltonian.drivingFields[0].amplitude.time_series.times[-1]
            )
        elif len(program.hamiltonian.shiftingFields) == 1:
            self.duration = float(
                program.hamiltonian.shiftingFields[0].magnitude.time_series.times[-1]
            )

        if self.duration == 0:
            self.simulation_times = [0]
        else:
            self.simulation_times = np.linspace(0, self.duration, steps)

        # Get valid configurations that comply with the blockade approximation
        self.atomArray = program.setup.ahs_register
        self.configurations = get_blockade_configurations(self.atomArray, self.blockade_radius)

        # Run the solver
        states = dm_scipy_integrate_ode_run(
            program,
            self.configurations,
            self.simulation_times,
            self.rydberg_interaction_coef,
            progress_bar=progress_bar,
            atol=atol,
            rtol=rtol,
            solver_method=solver_method,
            order=order,
            nsteps=nsteps,
            first_step=first_step,
            max_step=max_step,
            min_step=min_step,
            noises=noises,
        )

        # Convert the result type
        if shots == 0:
            raise NotImplementedError("shots=0 is not implemented")
        else:
            # convert density matrix to probabilities for post-processing
            dist = np.abs(np.diagonal(states[-1]))
            dist /= sum(dist)

            post_processed_info = apply_SPAM_noises(
                self.atomArray.filling, dist, self.configurations, noises
            )
            shot_results = sample_result(post_processed_info, shots, noises)
            return convert_result(
                shot_results,
                self._task_metadata(shots),
            )

    def _task_metadata(self, shots: int) -> TaskMetadata:
        return TaskMetadata(
            id="rydberg",
            shots=shots,
            deviceId="rydbergLocalSimulatorDM",
        )

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
