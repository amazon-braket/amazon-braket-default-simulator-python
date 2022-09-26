import time

import numpy as np
import scipy as sp
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import get_ops_coefs


def scipy_integrate_ode_run(
    hamiltonian: Program,
    configurations: list,
    simulation_times: list,
    rydberg_interaction_coef: float,
    progress_bar: bool = False,
    atol: bool = 1e-8,
    rtol: bool = 1e-6,
    solver_method: str = "adams",
    order: int = 12,
    nsteps: int = 1000,
    first_step: int = 0,
    max_step: int = 0,
    min_step: int = 0,
):
    """
    Using `scipy.integrate.ode` for solving the schrodinger equation

    Args:
        hamiltonian (Program): An analog simulation hamiltonian for Rydberg system
        configuraitons (list[str]): The list of configurations that comply with the
            blockade approximation.
        simulation_times (list[float]): The list of time points
        rydberg_interaction_coef (float): The interaction coefficient
        progress_bar (bool): If true, a progress bar will be printed during the simulation

        For the interpretations of the rest of the keyword arguments, see the document for
        `scipy.integrate.ode`
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html

    Return:
        states (List(np.ndarray)): The list of all the intermediate states in the simulation.


    """

    (
        rabi_ops,
        detuning_ops,
        local_detuning_ops,
        rabi_coefs,
        detuning_coefs,
        local_detuing_coefs,
        interaction_op,
    ) = get_ops_coefs(hamiltonian, configurations, rydberg_interaction_coef, simulation_times)

    print([len(simulation_times), len(rabi_ops), len(detuning_ops), len(local_detuning_ops)])

    def _get_hamiltonian(index_time):
        """Get the Hamiltonian matrix for the time point with index `index_time`"""
        index_time = int(index_time)
        h = interaction_op

        # Add the driving fields
        for rabi_op, rabi_coef, detuning_op, detuning_coef in zip(
            rabi_ops, rabi_coefs, detuning_ops, detuning_coefs
        ):
            h += (
                rabi_op * rabi_coef[index_time]
                + (rabi_op.T.conj() * np.conj(rabi_coef[index_time]))
                + detuning_op * detuning_coef[index_time]
            )

        # Add the shifting fields
        for local_detuning_op, local_detuning_coef in zip(local_detuning_ops, local_detuing_coefs):
            h += local_detuning_op * local_detuning_coef[index_time]

        return h

    # Define the initial state for the simulation
    size_hilbert_space = len(configurations)
    state = np.zeros(size_hilbert_space)
    state[0] = 1

    states = [state]  # The history of all intermediate states

    if len(simulation_times) == 1:
        return states

    dt = simulation_times[1] - simulation_times[0]  # The time step for the simulation

    def func(tind, y):
        return -1j * dt * _get_hamiltonian(tind).dot(y)

    r = sp.integrate.ode(func)
    r.set_integrator(
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

    for index_time in range(len(simulation_times) - 1):

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

        if not r.successful():
            raise Exception(
                "ODE integration error: Try to increase "
                "the allowed number of substeps by increasing "
                "the nsteps parameter in the Options class."
            )

        r.set_initial_value(state, index_time)
        r.integrate(index_time + 1)

        # get the current state, and normalize it
        state = r.y
        state /= np.linalg.norm(state)  # normalize the state
        states.append(state)

    return states
