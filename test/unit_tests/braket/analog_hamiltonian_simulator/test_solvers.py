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
local_detuning = {"magnitude": magnitude}
hamiltonian = {"drivingFields": [driving_field], "localDetuning": [local_detuning]}

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
    hamiltonian={"drivingFields": [], "localDetuning": []},
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


# Test solver with non-constant amplitude and detuning

# We use a single atom and set amplitude and detuning equal
# to each other, so that the evolution can be solved exactly

theta = np.pi / 2  # could be arbitrary value

amplitude_2 = {
    "pattern": "uniform",
    "time_series": {"times": [0, tmax / 2, tmax], "values": [0.0, theta / tmax, 0.0]},
}

detuning_2 = {
    "pattern": "uniform",
    "time_series": {"times": [0, tmax / 2, tmax], "values": [0.0, theta / tmax, 0.0]},
}

phase_2 = {"pattern": "uniform", "time_series": {"times": [0, tmax], "values": [0, 0]}}

driving_field_2 = {"amplitude": amplitude_2, "phase": phase_2, "detuning": detuning_2}

setup_2 = {"ahs_register": {"sites": [[0, 0]], "filling": [1]}}
hamiltonian_2 = {"drivingFields": [driving_field_2], "localDetuning": []}

program_2 = convert_unit(
    Program(
        setup=setup_2,
        hamiltonian=hamiltonian_2,
    )
)

theory_value_1 = (np.sin(1 / 2 * theta / np.sqrt(2)) / np.sqrt(2)) ** 2
theory_value_0 = 1 - theory_value_1

configurations_2 = ["g", "r"]
true_final_prob_2 = [theory_value_0, theory_value_1]


@pytest.mark.parametrize(
    "solver",
    [
        scipy_integrate_ode_run,
        rk_run,
    ],
)
def test_solvers_non_constant_program(solver):
    states = solver(
        program_2,
        configurations_2,
        simulation_times,
        rydberg_interaction_coef,
    )
    final_prob = [np.abs(i) ** 2 for i in states[-1]]

    assert np.allclose(final_prob, true_final_prob_2, atol=1e-2)


# Test a program with the following properties
# 1. It has vacant site and a local detuning field
# 2. The vacant site is added to the register before another filled site

# Define a global pi-pulse
driving_field_pi_pulse = {
    "amplitude": {
        "pattern": "uniform",
        "time_series": {"times": [0, tmax], "values": [np.pi / tmax, np.pi / tmax]},
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
# Case 1: the first site is filled, and the second site is empty
ahs_register_1 = {"sites": [[0, 0], [10e-4, 0], [20e-4, 0]], "filling": [1, 1, 0]}
pattern_1 = [0, 1, 1]

# Case 2: the first site is filled, and the second site is empty
# Note that the registers in case 1 and 2 are physically the same
# but with different labelings
ahs_register_2 = {"sites": [[20e-4, 0], [0, 0], [10e-4, 0]], "filling": [0, 1, 1]}
pattern_2 = [1, 0, 1]


# Define the programs for the two cases
def get_program_with_vacant_site_pi_pulse(
    ahs_register,
    pattern,
    shifting_field_time_series=shifting_field_time_series,
    driving_field_pi_pulse=driving_field_pi_pulse,
):
    shifting_field = {"magnitude": {"pattern": pattern, "time_series": shifting_field_time_series}}
    return convert_unit(
        Program(
            setup={"ahs_register": ahs_register},
            hamiltonian={
                "drivingFields": [driving_field_pi_pulse],
                "shiftingFields": [shifting_field],
            },
        )
    )


program_with_vacant_site_1 = get_program_with_vacant_site_pi_pulse(ahs_register_1, pattern_1)
program_with_vacant_site_2 = get_program_with_vacant_site_pi_pulse(ahs_register_2, pattern_2)

# Test the result. Upon inspection, we conclude that the state should be 'rg'
# for both cases.


@pytest.mark.parametrize(
    "solver, program",
    [
        [scipy_integrate_ode_run, program_with_vacant_site_1],
        [scipy_integrate_ode_run, program_with_vacant_site_2],
        [rk_run, program_with_vacant_site_1],
        [rk_run, program_with_vacant_site_2],
    ],
)
def test_program_with_vacant_site_pi_pulse(solver, program):
    states = solver(program, ["gg", "gr", "rg", "rr"], simulation_times, rydberg_interaction_coef)
    final_prob = [np.abs(i) ** 2 for i in states[-1]]
    true_final_prob = [0, 0, 1, 0]
    assert np.allclose(final_prob, true_final_prob, atol=1e-2)
