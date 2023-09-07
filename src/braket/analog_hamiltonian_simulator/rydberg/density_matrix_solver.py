import time
from typing import List

import numpy as np
import scipy.integrate
import scipy.sparse
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _apply_lindbladian,
    _get_ops_coefs,
    _get_ops_coefs_lind,
    _print_progress_bar,
)


def dm_scipy_integrate_ode_run(
    program: Program,
    configurations: List[str],
    simulation_times: List[float],
    rydberg_interaction_coef: float,
    progress_bar: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    solver_method: str = "adams",
    order: int = 12,
    nsteps: int = 1000,
    first_step: int = 0,
    max_step: int = 0,
    min_step: int = 0,
    noises: dict = None,
) -> np.ndarray:
    h0_operators_coefficients = _get_ops_coefs(
        program, configurations, rydberg_interaction_coef, simulation_times
    )

    lind_operators_coefficients = _get_ops_coefs_lind(
        np.arange(np.count_nonzero(program.setup.ahs_register.filling)), configurations, noises
    )

    # Define the initial state for the simulation
    size_hilbert_space = len(configurations)
    # the ode integrator only takes vector input
    state = np.zeros(size_hilbert_space * size_hilbert_space)
    state[0] = 1

    states = []
    # outputs are density matrices
    states.append(
        state.reshape((size_hilbert_space, size_hilbert_space))
    )  # The history of all intermediate states

    if len(simulation_times) == 1:
        return states

    dt = simulation_times[1] - simulation_times[0]  # The time step for the simulation

    # Define the function to be integrated, e.g. dy/dt = f(t, y).
    # Note that we we will use the index of the time point,
    # instead of time, for f(t, y).
    def f(index_time: int, y: np.ndarray) -> scipy.sparse.csr_matrix:
        return dt * _apply_lindbladian(
            index_time,
            h0_operators_coefficients,
            lind_operators_coefficients,
            y.reshape((size_hilbert_space, size_hilbert_space)),
        ).reshape((size_hilbert_space * size_hilbert_space, -1))

    integrator = scipy.integrate.ode(f)
    integrator.set_integrator(
        "zvode",
        atol=atol,
        rtol=rtol,
        method=solver_method,
        order=order,
        nsteps=nsteps,
        first_step=first_step,
        max_step=max_step,
        min_step=min_step,
    )

    start_time = time.time()
    for index_time, _ in enumerate(simulation_times[1:]):
        if progress_bar:  # print a lightweight progress bar
            _print_progress_bar(len(simulation_times), index_time, start_time)

        if not integrator.successful():
            raise Exception(
                "ODE integration error: Try to increase "
                "the allowed number of substeps by increasing "
                "the parameter `nsteps`."
            )

        integrator.set_initial_value(state, index_time)
        integrator.integrate(index_time + 1)

        # get the current vectorized state
        state = integrator.y
        # outputs are density matrices
        states.append(state.reshape((size_hilbert_space, size_hilbert_space)))

    return states
