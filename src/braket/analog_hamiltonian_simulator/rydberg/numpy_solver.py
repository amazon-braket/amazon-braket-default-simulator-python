import time
from typing import List

import numpy as np
import scipy.sparse
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import get_ops_coefs


def rk_run(
    hamiltonian: Program,
    configurations: List[str],
    simulation_times: List[float],
    rydberg_interaction_coef: float,
    progress_bar: bool = False,
) -> np.ndarray:
    """
    Implement the implicit Runge-Kutta method of order 6 for solving the schrodinger equation

    Args:
        hamiltonian (Program): An analog simulation hamiltonian for Rydberg system
        configurations (List[str]): The list of configurations that comply with the
            blockade approximation.
        simulation_times (List[float]): The list of time points
        rydberg_interaction_coef (float): The interaction coefficient
        progress_bar (bool): If true, a progress bar will be printed during the simulation.
            Default: False

    Returns:
        ndarray: The list of all the intermediate states in the simulation.

    Notes on the algorithm: For more details, please refer to
        https://en.wikipedia.org/wiki/Gauss-Legendre_method
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

    # define the Butcher tableau
    order = 6
    a = [
        [5 / 36, 2 / 9 - 1 / np.sqrt(15), 5 / 36 - np.sqrt(15) / 30],
        [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
        [5 / 36 + np.sqrt(15) / 30, 2 / 9 + 1 / np.sqrt(15), 5 / 36],
    ]
    b = [5 / 18, 4 / 9, 5 / 18]
    c = [1 / 2 - np.sqrt(15) / 10, 1 / 2, 1 / 2 + np.sqrt(15) / 10]

    stages = int(order / 2)  # The number of steps in the RK method see reference above

    eigvals_a, eigvecs_a = np.linalg.eig(a)
    inv_eigvecs_a = np.linalg.inv(eigvecs_a)

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

        x = states[-1]
        hamiltonian = _get_hamiltonian(index_time)

        # The start of implicit RK method for updating the state
        # For more details of the algorithm, see the reference above

        # Define k0,...,ks
        x1 = -1j * hamiltonian.dot(x)
        x2 = -1j * hamiltonian.dot(x1)
        x3 = -1j * hamiltonian.dot(x2)

        kk = [x1 + c[i] * dt * x2 for i in range(stages)]

        kx = [
            kk[i]
            - x1
            - dt * np.sum([a[i][j] * (x2 + c[j] * dt * x3) for j in range(stages)], axis=0)
            for i in range(stages)
        ]

        dk_tilde = [
            np.linalg.solve(
                np.eye(size_hilbert_space) + 1j * dt * eigvals_a[i] * hamiltonian,
                np.sum([inv_eigvecs_a[i][j] * kx[j] for j in range(stages)], axis=0),
            )
            for i in range(stages)
        ]

        dk = [
            np.sum([eigvecs_a[i][j] * dk_tilde[j] for j in range(stages)], axis=0)
            for i in range(stages)
        ]

        kk = np.array(kk) - dk

        delta_state = dt * np.array(b).dot(kk)  # The update of the state

        # The end of the implicit RK method for updating the state

        # Update the state, and save it
        state = x + delta_state
        states.append(state)

    return states
