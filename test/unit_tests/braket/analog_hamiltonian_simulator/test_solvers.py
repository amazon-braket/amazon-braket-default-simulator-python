import numpy as np
import pytest
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
)
from braket.analog_hamiltonian_simulator.rydberg.numpy_solver import rk_run
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    get_blockade_configurations,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)
from braket.analog_hamiltonian_simulator.rydberg.scipy_solver import scipy_integrate_ode_run

pi = np.pi
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF / ((SPACE_UNIT**6) / TIME_UNIT)

a = 3.0e-6
tmax = 0.25 * 1e-6  # note that we are in SI unites
rabi_frequency = 2 * pi * 4 * 1e6
detuning_value = 2 * pi * 3 * 1e6


amplitude = {
    "pattern": "uniform",
    "time_series": {"times": [0, tmax], "values": [rabi_frequency, rabi_frequency]},
}
detuning = {
    "pattern": "uniform",
    "time_series": {"times": [0, tmax], "values": [3 / 4 * detuning_value, 3 / 4 * detuning_value]},
}
phase = {"pattern": "uniform", "time_series": {"times": [0, tmax], "values": [0, 0]}}

driving_field = {"amplitude": amplitude, "phase": phase, "detuning": detuning}
magnitude = {
    "pattern": [1 / 4, 1 / 4],
    "time_series": {"times": [0, tmax], "values": [detuning_value, detuning_value]},
}
shifting_field = {"magnitude": magnitude}
hamiltonian = {"drivingFields": [driving_field], "shiftingFields": [shifting_field]}

setup = {"ahs_register": {"sites": [[0, 0], [0, a]], "filling": [1, 1]}}


program = convert_unit(
    Program(
        setup=setup,
        hamiltonian=hamiltonian,
    )
)

configurations = ["gg", "gr", "rg", "rr"]
steps = 400
simulation_times = np.linspace(0, tmax * 1e6, steps)


angular_frequency = np.sqrt(((detuning_value / 1e6) ** 2 + 8 * (rabi_frequency / 2 / 1e6) ** 2))
theory_value_01 = (
    4
    * (rabi_frequency / 2 / 1e6) ** 2
    * (np.sin(angular_frequency / 2 * (tmax * 1e6))) ** 2
    / (angular_frequency**2)
)
theory_value_10 = theory_value_01
theory_value_00 = 1 - theory_value_10 - theory_value_01


true_final_prob = [theory_value_00, theory_value_01, theory_value_10, 0]


@pytest.mark.parametrize(
    "solver, progress_bar",
    [
        [scipy_integrate_ode_run, True],
        [scipy_integrate_ode_run, False],
        [rk_run, True],
        [rk_run, False],
    ],
)
def test_solvers(solver, progress_bar):
    states = solver(
        program,
        configurations,
        simulation_times,
        rydberg_interaction_coef,
        progress_bar=progress_bar,
    )
    final_prob = [np.abs(i) ** 2 for i in states[-1]]

    assert np.allclose(final_prob, true_final_prob, atol=1e-2)


empty_program = Program(
    setup={
        "ahs_register": {
            "sites": [[0, i * a] for i in range(11)],
            "filling": [1 for _ in range(11)],
        }
    },
    hamiltonian={"drivingFields": [], "shiftingFields": []},
)

configurations_big_lattice = get_blockade_configurations(empty_program.setup.ahs_register, 0)

empty_program = convert_unit(empty_program)


@pytest.mark.parametrize(
    "solver, progress_bar",
    [
        [scipy_integrate_ode_run, True],
        [scipy_integrate_ode_run, False],
        [rk_run, True],
        [rk_run, False],
    ],
)
def test_solvers_empty_program(solver, progress_bar):
    states = solver(
        empty_program,
        configurations_big_lattice,
        [0],
        rydberg_interaction_coef,
        progress_bar=progress_bar,
    )
    final_prob = [np.abs(i) ** 2 for i in states[-1]]
    true_final_prob_empty_program = np.zeros(2**11)
    true_final_prob_empty_program[0] = 1

    assert np.allclose(final_prob, true_final_prob_empty_program, atol=1e-2)


failed_program = convert_unit(
    Program(
        setup={
            "ahs_register": {
                "sites": [[0, 0], [0, 1e-20]],
                "filling": [1, 1],
            }
        },
        hamiltonian=hamiltonian,
    )
)


def test_failed_scipy_run():
    try:
        scipy_integrate_ode_run(
            failed_program,
            configurations,
            simulation_times,
            rydberg_interaction_coef,
            progress_bar=True,
        )
    except Exception as e:
        assert str(e) == (
            "ODE integration error: Try to increase "
            "the allowed number of substeps by increasing "
            "the parameter `nsteps`."
        )

# Test a program with vacant site specified before a filled site
        
# Define a global pi-pulse
driving_field_pi_pulse = {
    "amplitude": {
        "pattern": "uniform",
        "time_series": {"times": [0, tmax], "values": [np.pi/tmax, np.pi/tmax]},
    },
    "phase": {
        "pattern": "uniform",
        "time_series": {"times": [0, tmax], "values": [0, 0]},
    },
    "detuning": {
        "pattern": "uniform",
        "time_series": {"times": [0, tmax], "values": [0, 0]},
    },
}

# Define the waveform for shifting field with large detuning values
shifting_field_time_series = {
    "times": [0, tmax], 
    "values": [1e10, 1e10],
}

# Define an atom arrangement with two different labelings
# where one of the site is filled  without LD, and the other
# site is empty and with strong LD

# Case 1: the first site is filled, and the second site is empty
ahs_register_1 = {"sites": [[0, 0], [10e-6, 0]], "filling": [1, 0]}

# Case 2: the first site is filled, and the second site is empty
# Note that the registers in case 1 and 2 are physically the same 
# but with different labelings
ahs_register_2 = {"sites": [[10e-6, 0], [0, 0]], "filling": [0, 1]}

# Define the programs for the two cases
# such that only the empty site experience the strong shifting field
def get_program_with_vacant_site_pi_pulse(
    ahs_register, 
    shifting_field_time_series = shifting_field_time_series,
    driving_field_pi_pulse = driving_field_pi_pulse
):
    pattern = list(1 - np.array(ahs_register["filling"]))
    shifting_field = {
        "magnitude": {
            "pattern": pattern,
            "time_series": shifting_field_time_series
        }
    }
    return convert_unit(
        Program(
            setup={"ahs_register": ahs_register},
            hamiltonian={
                "drivingFields": [driving_field_pi_pulse], 
                "shiftingFields": [shifting_field]
            },
        )
    )

program_1 = get_program_with_vacant_site_pi_pulse(ahs_register_1)
program_2 = get_program_with_vacant_site_pi_pulse(ahs_register_2)

# Test the result, we should see that the filled site is excited to 
# the Rydberg state

@pytest.mark.parametrize(
    "solver, program",
    [
        [scipy_integrate_ode_run, program_1],
        [scipy_integrate_ode_run, program_2],
        [rk_run, program_1],
        [rk_run, program_2]
    ],
)
def test_program_with_vacant_site_pi_pulse(solver, program):
    states = solver(program, ['g', 'r'], simulation_times, rydberg_interaction_coef)
    final_prob = [np.abs(i) ** 2 for i in states[-1]]
    true_final_prob = [0, 1]
    assert np.allclose(final_prob, true_final_prob, atol=1e-2)
