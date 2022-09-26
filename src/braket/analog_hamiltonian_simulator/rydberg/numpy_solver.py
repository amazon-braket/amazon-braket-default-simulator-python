import time

import numpy as np
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import get_ops_coefs


def RK_run(
    hamiltonian: Program,
    configurations: list,
    simulation_times: list,
    rydberg_interaction_coef: float,
    progress_bar: bool = False,
):
    """
    Implement the implicit Runge-Kutta method of order 6 for solving the schrodinger equation


    Args:
        hamiltonian (Program): An analog simulation hamiltonian for Rydberg system
        configuraitons (list[str]): The list of configurations that comply with the
            blockade approximation.
        simulation_times (list[float]): The list of time points
        rydberg_interaction_coef (float): The interaction coefficient
        progress_bar (bool): If true, a progress bar will be printed during the simulation

    Return:
        states (List(np.ndarray)): The list of all the intermediate states in the simulation.


    Notes on the algorithm:
        For more details, please refer to this link
        https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_method

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

    # define the Butcher tableau
    order = 6
    A = [
        [5 / 36, 2 / 9 - 1 / np.sqrt(15), 5 / 36 - np.sqrt(15) / 30],
        [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
        [5 / 36 + np.sqrt(15) / 30, 2 / 9 + 1 / np.sqrt(15), 5 / 36],
    ]
    b = [5 / 18, 4 / 9, 5 / 18]
    c = [1 / 2 - np.sqrt(15) / 10, 1 / 2, 1 / 2 + np.sqrt(15) / 10]

    s = int(order / 2)  # The number of steps in the RK method see reference above

    eigvals_A, eigvecs_A = np.linalg.eig(A)
    inv_eigvecs_A = np.linalg.inv(eigvecs_A)

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

        x = states[-1]
        ham = _get_hamiltonian(index_time)

        # The start of implicit RK method for updating the state
        # For more details of the algorithm, see the reference above

        # Define k0,...,ks
        x1 = -1j * ham.dot(x)
        x2 = -1j * ham.dot(x1)
        x3 = -1j * ham.dot(x2)

        kk = [x1 + c[ii] * dt * x2 for ii in range(s)]

        kx = [
            kk[ii]
            - x1
            - dt * np.sum([A[ii][jj] * (x2 + c[jj] * dt * x3) for jj in range(s)], axis=0)
            for ii in range(s)
        ]

        dk_tilde = [
            np.linalg.solve(
                np.eye(size_hilbert_space) + 1j * dt * eigvals_A[ii] * ham,
                np.sum([inv_eigvecs_A[ii][jj] * kx[jj] for jj in range(s)], axis=0),
            )
            for ii in range(s)
        ]

        dk = [
            np.sum([eigvecs_A[ii][jj] * dk_tilde[jj] for jj in range(s)], axis=0) for ii in range(s)
        ]

        kk = np.array(kk) - dk

        delta_state = dt * np.array(b).dot(kk)  # The update of the state

        # The end of the implicit RK method for updating the state

        # Update the state, and save it
        state = x + delta_state
        states.append(state)

    return states
