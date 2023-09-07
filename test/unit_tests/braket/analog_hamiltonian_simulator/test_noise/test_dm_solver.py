import numpy as np
import pytest
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
)
from braket.analog_hamiltonian_simulator.rydberg.density_matrix_solver import (
    dm_scipy_integrate_ode_run,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    get_blockade_configurations,
    noise_type,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

pi = np.pi
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF / ((SPACE_UNIT**6) / TIME_UNIT)

a = 3.0e-6
tmax = 0.25 * 1e-6  # note that we are in SI unites
rabi_frequency = 2 * pi * 4 * 1e6
detuning_value = 2 * pi * 3 * 1e6
local_detuning_value = 2 * pi * 2 * 1e6
phase = 3 / 5 * pi


amplitude = {
    "pattern": "uniform",
    "time_series": {"times": [0, tmax], "values": [rabi_frequency, rabi_frequency]},
}
detuning = {
    "pattern": "uniform",
    "time_series": {"times": [0, tmax], "values": [detuning_value, detuning_value]},
}
phase = {"pattern": "uniform", "time_series": {"times": [0, tmax], "values": [phase, phase]}}

driving_field = {"amplitude": amplitude, "phase": phase, "detuning": detuning}
magnitude = {
    "pattern": [1 / 4, 3 / 4],
    "time_series": {"times": [0, tmax], "values": [local_detuning_value, local_detuning_value]},
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
steps = 100
simulation_times = np.linspace(0, tmax * 1e6, steps)

noises = {
    noise_type.T_1: 100,
    noise_type.T_2: 25,
}

true_dm_mid = np.array(
    [
        [
            8.39695176e-01 - 4.58918326e-17j,
            -1.30194285e-01 + 9.43682358e-02j,
            -3.06075197e-01 - 1.04649944e-01j,
            -2.49973331e-04 - 7.10957459e-04j,
        ],
        [
            -1.30194285e-01 - 9.43682358e-02j,
            3.27048254e-02 + 9.70808139e-18j,
            3.52013396e-02 + 5.05041007e-02j,
            -3.98737700e-05 + 1.40283044e-04j,
        ],
        [
            -3.06075197e-01 + 1.04649944e-01j,
            3.52013396e-02 - 5.05041007e-02j,
            1.27118965e-01 + 3.55853427e-17j,
            1.80864886e-04 + 2.30828956e-04j,
        ],
        [
            -2.49973331e-04 + 7.10957459e-04j,
            -3.98737700e-05 - 1.40283044e-04j,
            1.80864886e-04 - 2.30828956e-04j,
            4.81033482e-04 + 5.05318076e-20j,
        ],
    ]
)
true_dm_final = np.array(
    [
        [
            6.48951696e-01 + 1.55199598e-16j,
            -3.60697796e-01 + 3.90234045e-02j,
            -1.92285721e-01 - 2.28788020e-01j,
            -1.33331012e-05 - 1.06210186e-03j,
        ],
        [
            -3.60697796e-01 - 3.90234045e-02j,
            2.07475290e-01 - 1.96292725e-16j,
            9.29206128e-02 + 1.38940491e-01j,
            -5.50212408e-05 + 6.00651945e-04j,
        ],
        [
            -1.92285721e-01 + 2.28788020e-01j,
            9.29206128e-02 - 1.38940491e-01j,
            1.42645286e-01 + 4.02141458e-17j,
            3.81300764e-04 + 3.17095126e-04j,
        ],
        [
            -1.33331012e-05 + 1.06210186e-03j,
            -5.50212408e-05 - 6.00651945e-04j,
            3.81300764e-04 - 3.17095126e-04j,
            9.27729022e-04 + 6.19795594e-20j,
        ],
    ]
)


def test_dm_solver():
    states = dm_scipy_integrate_ode_run(
        program, configurations, simulation_times, rydberg_interaction_coef, noises=noises
    )
    state_mid = states[49]
    state_final = states[-1]
    assert np.allclose(state_mid, true_dm_mid, atol=1e-2)
    assert np.allclose(state_final, true_dm_final, atol=1e-2)


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
        [dm_scipy_integrate_ode_run, True],
        [dm_scipy_integrate_ode_run, False],
    ],
)
def test_dm_solvers_empty_program(solver, progress_bar):
    states = solver(
        empty_program,
        configurations_big_lattice,
        [0],
        rydberg_interaction_coef,
        progress_bar=progress_bar,
        noises=noises,
    )
    final_dm = states[-1]
    true_final_dm = np.zeros((2**11, 2**11))
    true_final_dm[0, 0] = 1

    assert np.allclose(final_dm, true_final_dm, atol=1e-2)


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
        dm_scipy_integrate_ode_run(
            failed_program,
            configurations,
            simulation_times,
            rydberg_interaction_coef,
            progress_bar=True,
            noises=noises,
        )
    except Exception as e:
        assert str(e) == (
            "ODE integration error: Try to increase "
            "the allowed number of substeps by increasing "
            "the parameter `nsteps`."
        )
