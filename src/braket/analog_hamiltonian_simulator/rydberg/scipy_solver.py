import time
from typing import List

import numpy as np
import scipy.integrate
import scipy.sparse
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import get_ops_coefs


def scipy_integrate_ode_run(
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
) -> np.ndarray:
    """
    Solves the Schrödinger equation with `scipy.integrate.ode`

    Args:
        program (Program): An analog program for the Rydberg system to be simulated
        configurations (List[str]): The list of configurations that comply with the
            blockade approximation.
        simulation_times (List[float]): The list of time points
        rydberg_interaction_coef (float): The interaction coefficient
        progress_bar (bool): If true, a progress bar will be printed during the simulation.
            Default: False
        atol (float): Absolute tolerance for solution. Default: 1e-8
        rtol (float): Relative tolerance for solution. Default: 1e-6
        solver_method (str): Which solver to use, `adams` for non-stiff problems or `bdf`
            for stiff problems. Default: "adams"
        order (int): Maximum order used by the integrator, order <= 12 for Adams, <= 5 for BDF.
            Default: 12
        nsteps (int): Maximum number of (internally defined) steps allowed during one call to
            the solver. Default: 1000
        first_step (int): Default: 0
        max_step (int): Limits for the step sizes used by the integrator. Default: 0
        min_step (int): Default: 0

    Returns:
        ndarray: The list of all the intermediate states in the simulation.

    For more information, please refer to the documentation for `scipy.integrate.ode`
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
        
    """

    (
        rabi_ops,
        detuning_ops,
        local_detuning_ops,
        rabi_coefs,
        detuning_coefs,
        local_detuing_coefs,
        interaction_op,
    ) = get_ops_coefs(program, configurations, rydberg_interaction_coef, simulation_times)

    def _get_hamiltonian(index_time: int) -> scipy.sparse.csr_matrix:
        """Get the Hamiltonian matrix for the time point with index `index_time`"""
        index_time = int(index_time)
        hamiltonian = interaction_op

        # Add the driving fields
        for rabi_op, rabi_coef, detuning_op, detuning_coef in zip(
            rabi_ops, rabi_coefs, detuning_ops, detuning_coefs
        ):
            hamiltonian += (
                rabi_op * rabi_coef[index_time] / 2
                + (rabi_op.T.conj() * np.conj(rabi_coef[index_time]) / 2)
                - detuning_op * detuning_coef[index_time]
            )

        # Add the shifting fields
        for local_detuning_op, local_detuning_coef in zip(local_detuning_ops, local_detuing_coefs):
            hamiltonian -= local_detuning_op * local_detuning_coef[index_time]

        return hamiltonian

    # Define the initial state for the simulation
    size_hilbert_space = len(configurations)
    state = np.zeros(size_hilbert_space)
    state[0] = 1

    states = [state]  # The history of all intermediate states

    if len(simulation_times) == 1:
        return states

    dt = simulation_times[1] - simulation_times[0]  # The time step for the simulation

    # Define the function to be integrated, e.g. dy/dt = f(t, y).
    # Note that we we will use the index of the time point,
    # instead of time, for f(t, y).
    def f(index_time: int, y: np.ndarray) -> scipy.sparse.csr_matrix:
        return -1j * dt * _get_hamiltonian(index_time).dot(y)

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

    for index_time, _ in enumerate(simulation_times[1:]):

        if progress_bar:  # print a lightweight progress bar
            if index_time == 0:
                start_time = time.time()
                print("0% finished, elapsed time = NA, ETA = NA", flush=True, end="\r")
            else:
                current_time = time.time()
                estimate_time_arrival = (
                    (current_time - start_time)
                    / (index_time + 1)
                    * (len(simulation_times) - (index_time + 1))
                )
                print(
                    f"{100 * (index_time+1)/len(simulation_times)}% finished, "
                    f"elapsed time = {(current_time-start_time)} seconds, "
                    f"ETA = {estimate_time_arrival} seconds ",
                    flush=True,
                    end="\r",
                )

        if not integrator.successful():
            raise Exception(
                "ODE integration error: Try to increase "
                "the allowed number of substeps by increasing "
                "the nsteps parameter in the Options class."
            )

        integrator.set_initial_value(state, index_time)
        integrator.integrate(index_time + 1)

        # get the current state, and normalize it
        state = integrator.y
        state /= np.linalg.norm(state)  # normalize the state
        states.append(state)

    return states
