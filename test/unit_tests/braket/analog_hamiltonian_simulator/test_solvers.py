import numpy as np
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
)
from braket.analog_hamiltonian_simulator.rydberg.numpy_solver import RK_run
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)
from braket.analog_hamiltonian_simulator.rydberg.scipy_solver import scipy_integrate_ode_run

pi = np.pi
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF / ((SPACE_UNIT**6) / TIME_UNIT)

a = 3.0e-6
tmax = 0.25 * 1e-6  # note that we are in SI unites
Omega = 2 * pi * 4 * 1e6
Delta = 2 * pi * 3 * 1e6


amplitude = {"pattern": "uniform", "sequence": {"times": [0, tmax], "values": [Omega, Omega]}}
detuning = {
    "pattern": "uniform",
    "sequence": {"times": [0, tmax], "values": [3 / 4 * Delta, 3 / 4 * Delta]},
}
phase = {"pattern": "uniform", "sequence": {"times": [0, tmax], "values": [0, 0]}}

driving_field = {"amplitude": amplitude, "phase": phase, "detuning": detuning}
magnitude = {"pattern": [1 / 4, 1 / 4], "sequence": {"times": [0, tmax], "values": [Delta, Delta]}}
shifting_field = {"magnitude": magnitude}
hamiltonian = {"drivingFields": [driving_field], "shiftingFields": [shifting_field]}

setup = {"atomArray": {"sites": [[0, 0], [0, a]], "filling": [1, 1]}}


program = convert_unit(
    Program(
        setup=setup,
        hamiltonian=hamiltonian,
    )
)

configurations = ["gg", "gr", "rg", "rr"]
steps = 400
simulation_times = np.linspace(0, tmax * 1e6, steps)


ω = np.sqrt(((Delta / 1e6) ** 2 + 8 * (Omega / 1e6) ** 2))
theory_value_01 = 4 * (Omega / 1e6) ** 2 * (np.sin(ω / 2 * (tmax * 1e6))) ** 2 / (ω**2)
theory_value_10 = theory_value_01
theory_value_00 = 1 - theory_value_10 - theory_value_01


true_final_prob = [theory_value_00, theory_value_01, theory_value_10, 0]


def test_scipy_integrate_ode_run():
    states = scipy_integrate_ode_run(
        program,
        configurations,
        simulation_times,
        rydberg_interaction_coef,
        progress_bar=True,
    )
    final_prob = [np.abs(i) ** 2 for i in states[-1]]

    assert np.allclose(final_prob, true_final_prob, atol=1e-2)


def test_RK_run():
    states = RK_run(
        program,
        configurations,
        simulation_times,
        rydberg_interaction_coef,
        progress_bar=True,
    )
    final_prob = [np.abs(i) ** 2 for i in states[-1]]

    assert np.allclose(final_prob, true_final_prob, atol=1e-2)
