import numpy as np
from braket.device_schema import DeviceCapabilities
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationTaskResult,
)
from braket.task_result.task_metadata_v1 import TaskMetadata

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
    capabilities_constants,
)
from braket.analog_hamiltonian_simulator.rydberg.numpy_solver import RK_run
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

from pydantic import create_model  # This is temporary for defining properties below
import sys

class RydbergAtomSimulator(BaseLocalSimulator):
    DEVICE_ID = "rydberg"

    def run(
        self,
        program: Program,
        shots: int = 1000,
        steps: int = 100,
        rydberg_interaction_coef: float = RYDBERG_INTERACTION_COEF,
        blockade_radius: float = 0.0,
        progress_bar: bool = False,
        atol: bool = 1e-8,
        rtol: bool = 1e-6,
        solver_method: str = "adams",
        order: int = 12,
        nsteps: int = 1000,
        first_step: int = 0,
        max_step: int = 0,
        min_step: int = 0,
    ) -> AnalogHamiltonianSimulationTaskResult:
        """
        Run the simulation for the dynamics of the Rydberg system, and return the result

        Args:
            program (Program): An analog simulation hamiltonian for Rydberg system
            shots (int): The number of shots
            steps (int): The number of time points for the simulation
            rydberg_interaction_coef (float): The interaction coefficient
            blockade_radius (float): The blockade radius for the Rydberg system
            progress_bar (bool): If true, a progress bar will be printed during the simulation

            For the interpretations of the rest of the keyword arguments, see the document for
            `scipy.integrate.ode`
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html.

            Note that for system with the size of Hilbert space less than 1000,
            these keyword arguments do not apply.


        Return:
            (AnalogHamiltonianSimulationTaskResult): The result of the simulation

        """

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
            self.duration = float(program.hamiltonian.drivingFields[0].amplitude.sequence.times[-1])
        elif len(program.hamiltonian.shiftingFields) == 1:
            self.duration = float(
                program.hamiltonian.shiftingFields[0].magnitude.sequence.times[-1]
            )

        if self.duration == 0:
            self.simulation_times = [0]
        else:
            self.simulation_times = np.linspace(0, self.duration, steps)

        # Get valid configurations that comply with the blockade approximation
        self.atomArray = program.setup.atomArray
        self.configurations = get_blockade_configurations(self.atomArray, self.blockade_radius)

        # Run the solver
        # We shall adaptive change between numpy solver (RK6 method) and scipy solver
        if len(self.configurations) <= 1000:
            states = RK_run(
                program,
                self.configurations,
                self.simulation_times,
                self.rydberg_interaction_coef,
            )
        else:
            states = scipy_integrate_ode_run(
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
            )

        # Convert the result type
        if shots == 0:
            raise NotImplementedError("Shot = 0 is not implemented yet")
        else:
            dist = sample_state(states[-1], shots)
            return convert_result(
                dist,
                self.atomArray.filling,
                self.configurations,
                self.mock_taskMetadata(shots),
            )

    def mock_taskMetadata(self, shots):
        return TaskMetadata(
            id="rydberg",
            shots=shots,
            deviceId="rydbergLocalSimulator",
        )

    @property
    def properties(self) -> DeviceCapabilities:
        """simulator properties"""
        mock_dict = {
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
            "RydbergSimulatorDeviceCapabilities", **mock_dict
        )

        return RydbergSimulatorDeviceCapabilities.parse_obj(mock_dict)                    

    def initialize_simulation(self, **kwargs) -> Simulation:
        """Initializes simulation with keyword arguments"""
        pass
