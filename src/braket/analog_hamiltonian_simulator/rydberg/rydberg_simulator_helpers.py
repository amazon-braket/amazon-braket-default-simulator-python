import functools as ft
import itertools
import time
import warnings
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse
from braket.ir.ahs.atom_arrangement import AtomArrangement
from braket.ir.ahs.program_v1 import Program


class noise_type(Enum):
    ATOM_DETECTION = "atom_detection"
    VACANCY_DETECTION = "vacancy_detection"
    GROUND_STATE_DETECTION = "ground_state_detection"
    RYDBERG_STATE_DETECTION = "rydberg_state_detection"
    T_1 = "T_1"
    T_2 = "T_2"


def validate_config(config: str, atoms_coordinates: np.ndarray, blockade_radius: float) -> bool:
    """Valid if a given configuration complies with the Rydberg approximation

    Args:
        config (str): The configuration to be validated
        atoms_coordinates (ndarray): The coordinates for atoms in the filled sites
        blockade_radius (float): The Rydberg blockade radius

    Returns:
        bool: True if the configuration complies with the Rydberg approximation,
        False otherwise
    """

    # The indices for the Rydberg atoms in the configuration
    rydberg_atoms = [i for i, item in enumerate(config) if item == "r"]

    for i, rydberg_atom in enumerate(rydberg_atoms[:-1]):
        dists = np.linalg.norm(
            atoms_coordinates[rydberg_atom] - atoms_coordinates[rydberg_atoms[i + 1 :]], axis=1
        )
        if min(dists) <= blockade_radius:
            return False
    return True


def get_blockade_configurations(lattice: AtomArrangement, blockade_radius: float) -> List[str]:
    """Return the lattice configurations complying with the blockade approximation

    Args:
        lattice (AtomArrangement): A lattice with Rydberg atoms and their coordinates
        blockade_radius (float): The Rydberg blockade radius

    Returns:
        List[str]: A list of bit strings, each of them corresponding to a valid
        configuration complying with the blockade approximation. The length of
        each configuration is the same as the number of atoms in the lattice,
        with 'r' and 'g' indicating the Rydberg and ground states, respectively.

        Notes on the indexing: The left-most bit in the configuration corresponds to
        the first atom in the lattice.

        Notes on the algorithm: We start from all possible configurations and get rid of
        those violating the blockade approximation constraint.
    """

    # The coordinates for atoms in the filled sites
    atoms_coordinates = np.array(lattice.sites)[np.where(lattice.filling)]
    min_separation = float("inf")  # The minimum separation between atoms, or filled sites
    for i, atom_coord in enumerate(atoms_coordinates[:-1]):
        dists = np.linalg.norm(atom_coord - atoms_coordinates[i + 1 :], axis=1)
        min_separation = min(min_separation, min(dists))

    configurations = [
        "".join(item) for item in itertools.product(["g", "r"], repeat=sum(lattice.filling))
    ]

    if blockade_radius < min_separation:  # no need to consider blockade approximation
        return configurations
    return [
        config
        for config in configurations
        if validate_config(config, atoms_coordinates, blockade_radius)
    ]


def _get_interaction_dict(
    program: Program, rydberg_interaction_coef: float, configurations: List[str]
) -> Dict[Tuple[int, int], float]:
    """Return the dict contains the Rydberg interaction strength for all configurations.

    Args:
        program (Program): An analog simulation program for Rydberg system with the interaction term
        rydberg_interaction_coef (float): The interaction coefficient
        configurations (List[str]): The list of configurations that comply with the blockade
            approximation.

    Returns:
        Dict[Tuple[int, int], float]: The dictionary for the interaction operator
    """

    # The coordinates for atoms in the filled sites
    lattice = program.setup.ahs_register
    atoms_coordinates = np.array(
        [lattice.sites[i] for i in range(len(lattice.sites)) if lattice.filling[i] == 1]
    )

    interactions = {}  # The interaction in the basis of configurations, as a dictionary

    for config_index, config in enumerate(configurations):
        interaction = 0

        # The indices for the Rydberg atoms in the configuration
        rydberg_atoms = [i for i, item in enumerate(config) if item == "r"]

        # Obtain the pairwise distances between the Rydberg atoms, followed by adding their Rydberg
        # interactions
        for ind_1, rydberg_atom_1 in enumerate(rydberg_atoms[:-1]):
            for ind_2, rydberg_atom_2 in enumerate(rydberg_atoms):
                if ind_2 > ind_1:
                    dist = np.linalg.norm(
                        atoms_coordinates[rydberg_atom_1] - atoms_coordinates[rydberg_atom_2]
                    )
                    interaction += rydberg_interaction_coef / (float(dist) ** 6)

        if interaction > 0:
            interactions[(config_index, config_index)] = interaction

    return interactions


def _get_detuning_dict(
    targets: Tuple[int], configurations: List[str]
) -> Dict[Tuple[int, int], float]:
    """Return the dict contains the detuning operators for a set of target atoms.

    Args:
        targets (Tuple[int]): The target atoms of the detuning operator
        configurations (List[str]): The list of configurations that comply with the blockade
            approximation.

    Returns:
        Dict[Tuple[int, int], float]: The dictionary for the detuning operator
    """

    detuning = {}  # The detuning term in the basis of configurations, as a dictionary

    for ind_1, config in enumerate(configurations):
        value = sum([1 for ind_2, item in enumerate(config) if item == "r" and ind_2 in targets])
        if value > 0:
            detuning[(ind_1, ind_1)] = value

    return detuning


def _get_rabi_dict(targets: Tuple[int], configurations: List[str]) -> Dict[Tuple[int, int], float]:
    """Return the dict for the Rabi operators for a set of target atoms.

    Args:
        targets (Tuple[int]): The target atoms of the detuning operator
        configurations (List[str]): The list of configurations that comply with the blockade
            approximation.

    Returns:
        Dict[Tuple[int, int], float]: The dictionary for the Rabi operator

    Note:
        We only save the lower triangular part of the matrix that corresponds
        to the Rabi operator.
    """

    rabi = {}  # The Rabi term in the basis of configurations, as a dictionary

    # use dictionary to store index of configurations
    configuration_index = {config: ind for ind, config in enumerate(configurations)}

    for ind_1, config_1 in enumerate(configurations):
        for target in targets:
            # Only keep the lower triangular part of the Rabi operator
            # which convert a single atom from "g" to "r".
            if config_1[target] != "g":
                continue

            # Construct the state after applying the Rabi operator
            bit_list = list(config_1)
            bit_list[target] = "r"
            config_2 = "".join(bit_list)

            # If the constructed state is in the Hilbert space,
            # add the corresponding matrix element to the Rabi operator.
            if config_2 in configuration_index:
                rabi[(configuration_index[config_2], ind_1)] = 1

    return rabi


def _get_sparse_from_dict(
    matrix_dict: Dict[Tuple[int, int], float], matrix_dimension: int
) -> scipy.sparse.csr_matrix:
    """Convert a dict to a CSR sparse matrix

    Args:
        matrix_dict (Dict[Tuple[int, int], float]): The dict for the sparse matrix
        matrix_dimension (int): The size of the sparse matrix

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix in CSR format
    """
    rows = [key[0] for key in matrix_dict.keys()]
    cols = [key[1] for key in matrix_dict.keys()]
    return scipy.sparse.csr_matrix(
        tuple([list(matrix_dict.values()), [rows, cols]]),
        shape=(matrix_dimension, matrix_dimension),
    )


def _get_sparse_ops(
    program: Program, configurations: List[str], rydberg_interaction_coef: float
) -> Tuple[
    List[scipy.sparse.csr_matrix],
    List[scipy.sparse.csr_matrix],
    scipy.sparse.csr_matrix,
    List[scipy.sparse.csr_matrix],
]:
    """Returns the sparse matrices for Rabi, detuning, interaction and local detuning detuning
    operators

    Args:
        program (Program): An analog simulation program for Rydberg system
        configurations (List[str]): The list of configurations that comply with the blockade
            approximation.
        rydberg_interaction_coef (float): The interaction coefficient

    Returns:
        Tuple[List[csr_matrix],List[csr_matrix],csr_matrix,List[csr_matrix]]: A tuple containing
        the list of Rabi operators, the list of detuing operators,
        the interaction operator and the list of local detuing operators

    """
    # Get the driving fields as sparse matrices, whose targets are all the atoms in the system
    targets = np.arange(np.count_nonzero(program.setup.ahs_register.filling))
    rabi_dict = _get_rabi_dict(targets, configurations)
    detuning_dict = _get_detuning_dict(targets, configurations)

    # Driving field is an array of operators, which has only one element for now
    rabi_ops = [_get_sparse_from_dict(rabi_dict, len(configurations))]
    detuning_ops = [_get_sparse_from_dict(detuning_dict, len(configurations))]

    # Get the interaction term as a sparse matrix
    interaction_dict = _get_interaction_dict(program, rydberg_interaction_coef, configurations)
    interaction_op = _get_sparse_from_dict(interaction_dict, len(configurations))

    # Get the shifting fields as sparse matrices.
    # Shifting field is an array of operators, which has only one element for now
    local_detuning_ops = []
    for shifting_field in program.hamiltonian.shiftingFields:
        temp = 0
        for site in range(len(shifting_field.magnitude.pattern)):
            strength = shifting_field.magnitude.pattern[site]
            opt = _get_sparse_from_dict(
                _get_detuning_dict((site,), configurations), len(configurations)
            )
            temp += float(strength) * scipy.sparse.csr_matrix(opt, dtype=float)

        local_detuning_ops.append(temp)

    return rabi_ops, detuning_ops, interaction_op, local_detuning_ops


def _interpolate_time_series(
    t: float, times: List[float], values: List[float], method: str = "piecewise_linear"
) -> float:
    """Interpolates the value of a series of time-value pairs at the given time via linear
        interpolation.

    Args:
        t (float): The given time point
        times (List[float]): The list of time points
        values (List[float]): The list of values
        method (str): The method for interpolation, either "piecewise_linear" or
            "piecewise_constant." Default: "piecewise_linear"

    Returns:
        float: The interpolated value of the time series at t
    """

    times = [float(time) for time in times]
    values = [float(value) for value in values]

    if method == "piecewise_linear":
        return np.interp(t, times, values)
    elif method == "piecewise_constant":
        index = np.searchsorted(times, t, side="right") - 1
        return values[index]
    else:
        raise ValueError("`method` can only be `piecewise_linear` or `piecewise_constant`.")


def _get_coefs(
    program: Program, simulation_times: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients for the Rabi operators, detuning operators and local detuning
    operators for all the time points in the analog simulation program.

    Args:
        program (Program): An analog simulation program for Rydberg system
        simulation_times (List[float]): The list of time points

    Returns:
        Tuple[ndarray, ndarray, ndarray]: A tuple containing
        the list of Rabi frequencies, the list of global detuings and
        the list of local detunings
    """
    rabi_coefs, detuning_coefs = [], []

    for driving_field in program.hamiltonian.drivingFields:
        amplitude = driving_field.amplitude.time_series
        phase = driving_field.phase.time_series
        detuning = driving_field.detuning.time_series

        # Get the Rabi part. We use the convention: Omega * exp(1j*phi) * |r><g| + h.c.
        rabi_coef = np.array(
            [
                _interpolate_time_series(
                    t, amplitude.times, amplitude.values, method="piecewise_linear"
                )
                * np.exp(
                    1j
                    * _interpolate_time_series(
                        t, phase.times, phase.values, method="piecewise_constant"
                    )
                )
                for t in simulation_times
            ],
            dtype=complex,
        )
        rabi_coefs.append(rabi_coef)

        # Get the detuning part
        detuning_coef = np.array(
            [
                _interpolate_time_series(
                    t, detuning.times, detuning.values, method="piecewise_linear"
                )
                for t in simulation_times
            ],
            dtype=complex,
        )
        detuning_coefs.append(detuning_coef)

    # add shifting fields
    local_detuing_coefs = []
    for shifting_field in program.hamiltonian.shiftingFields:
        magnitude = shifting_field.magnitude.time_series

        local_detuing_coef = np.array(
            [
                _interpolate_time_series(
                    t, magnitude.times, magnitude.values, method="piecewise_linear"
                )
                for t in simulation_times
            ],
            dtype=complex,
        )
        local_detuing_coefs.append(local_detuing_coef)

    return np.array(rabi_coefs), np.array(detuning_coefs), np.array(local_detuing_coefs)


def _get_ops_coefs(
    program: Program,
    configurations: List[str],
    rydberg_interaction_coef: float,
    simulation_times: List[float],
) -> Tuple[
    List[scipy.sparse.csr_matrix],
    List[scipy.sparse.csr_matrix],
    List[scipy.sparse.csr_matrix],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    scipy.sparse.csr_matrix,
]:
    """Returns the sparse matrices and coefficients for the Rabi terms, detuning terms and
    the local detuining terms, together with the interaction operator in the given analog
    simulation program for Rydberg systems.

    Args:
        program (Program): An analog simulation program for Rydberg system
        configurations (List[str]): The list of configurations that comply to the
            blockade approximation.
        rydberg_interaction_coef (float): The interaction coefficient
        simulation_times (List[float]): The list of time points

    Returns:
        Tuple[
            List[csr_matrix],
            List[csr_matrix],
            List[csr_matrix],
            ndarray,
            ndarray,
            ndarray,
            csr_matrix
        ]: A tuple containing the list of Rabi operators, the list of detuing operators,
        the list of local detuing operators, the list of Rabi frequencies, the list of global
        detuings, the list of local detunings and the interaction operator.
    """

    rabi_ops, detuning_ops, interaction_op, local_detuning_ops = _get_sparse_ops(
        program, configurations, rydberg_interaction_coef
    )
    rabi_coefs, detuning_coefs, local_detuing_coefs = _get_coefs(program, simulation_times)

    return (
        rabi_ops,
        detuning_ops,
        local_detuning_ops,
        rabi_coefs,
        detuning_coefs,
        local_detuing_coefs,
        interaction_op,
    )


# TODO: consider creating a class for result post-processing and sampling
def sample_result(
    post_processed_info: Tuple[np.ndarray, np.ndarray, List[int], List[int], Union[float, int]],
    shots: int,
    noises: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int],]:
    """Sample measurement outcomes from the post-processed distributions

    Args:
        post_processed_info (Tuple[
            np.ndarray,
            np.ndarray,
            List[int],
            List[int],
            Union[float, int]
            ]):  A tuple containing the post-processed pre-sequence distribution and
            state distribution, index of filled sites and empty sites and vacancy detection
            error rate.
        shots (int): The number of samples
        noises (Dict[str, float], optional): A dictionary of noises and their parameters.

    Returns:
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            List[int],
            List[int],
            ]: A tuple containing the sampled pre-sequences and vacant pose-sequences,
            state frequencies and index of filled sites and empty sites.
    """
    (pre_seq_dist, post_seq_dist, non_empty_sites, empty_sites, eps_vd) = post_processed_info

    num_qubits = len(non_empty_sites)
    num_vacancies = len(empty_sites)

    if (noise_type.ATOM_DETECTION in noises) or (noise_type.VACANCY_DETECTION in noises):
        pre_sequences = np.random.binomial(
            1, pre_seq_dist, size=(shots, num_qubits + num_vacancies)
        )
    else:
        pre_sequences = pre_seq_dist.reshape((1, -1))

    if noise_type.VACANCY_DETECTION in noises:
        post_sequences_empty = np.random.binomial(1, eps_vd, size=(shots, num_vacancies))
    else:
        post_sequences_empty = np.zeros(num_vacancies).reshape((1, -1))

    state_freq = np.random.multinomial(shots, post_seq_dist)

    return (
        pre_sequences,
        post_sequences_empty,
        state_freq,
        non_empty_sites,
        empty_sites,
    )


def _print_progress_bar(num_time_points: int, index_time: int, start_time: float) -> None:
    """Print a lightweight progress bar

    Args:
        num_time_points (int): The total number of time points
        index_time (int): The index of the current time point
        start_time (float): The starting time for the simulation

    """
    if index_time == 0:
        print("0% finished, elapsed time = NA, ETA = NA", flush=True, end="\r")
    else:
        current_time = time.time()
        estimate_time_arrival = (
            (current_time - start_time) / (index_time + 1) * (num_time_points - (index_time + 1))
        )
        print(
            f"{100 * (index_time+1)/num_time_points}% finished, "
            f"elapsed time = {(current_time-start_time)} seconds, "
            f"ETA = {estimate_time_arrival} seconds ",
            flush=True,
            end="\r",
        )


def _get_hamiltonian(
    index_time: float,
    operators_coefficients: Tuple[
        List[scipy.sparse.csr_matrix],
        List[scipy.sparse.csr_matrix],
        List[scipy.sparse.csr_matrix],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        scipy.sparse.csr_matrix,
    ],
) -> scipy.sparse.csr_matrix:
    """Get the Hamiltonian at a given time point

    Args:
        index_time (float): The index of the current time point
        operators_coefficients (Tuple[
            List[csr_matrix],
            List[csr_matrix],
            List[csr_matrix],
            ndarray,
            ndarray,
            ndarray,
            csr_matrix
        ]): A tuple containing the list of Rabi operators, the list of detuing operators,
        the list of local detuing operators, the list of Rabi frequencies, the list of global
        detuings, the list of local detunings and the interaction operator.

    Returns:
        (scipy.sparse.csr_matrix): The Hamiltonian at the given time point as a sparse matrix
    """
    (
        rabi_ops,
        detuning_ops,
        local_detuning_ops,
        rabi_coefs,
        detuning_coefs,
        local_detuing_coefs,
        interaction_op,
    ) = operators_coefficients

    index_time = int(index_time)

    if len(rabi_coefs) > 0:
        # If there is driving field, the maximum of index_time is the maximum time index
        # for the driving field.
        # Note that, if there is more than one driving field, we assume that they have the
        # same number of coefficients
        max_index_time = len(rabi_coefs[0]) - 1
    else:
        # If there is no driving field, then the maxium of index_time is the maxium time
        # index for the shifting field.
        # Note that, if there is more than one shifting field, we assume that they have the
        # same number of coefficients
        # Note that, if there is no driving field nor shifting field, the initial state will
        # be returned, and the simulation would not reach here.
        max_index_time = len(local_detuing_coefs[0]) - 1

    # If the integrator uses intermediate time value that is larger than the maximum
    # time value specified, the final time value is used as an approximation.
    if index_time > max_index_time:
        index_time = max_index_time
        warnings.warn(
            "The solver uses intermediate time value that is "
            "larger than the maximum time value specified. "
            "The final time value of the specified range "
            "is used as an approximation."
        )

    # If the integrator uses intermediate time value that is larger than the minimum
    # time value specified, the final time value is used as an approximation.
    if index_time < 0:
        index_time = 0
        warnings.warn(
            "The solver uses intermediate time value that is "
            "smaller than the minimum time value specified. "
            "The first time value of the specified range "
            "is used as an approximation."
        )

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


def _apply_hamiltonian(
    index_time: float,
    operators_coefficients: Tuple[
        List[scipy.sparse.csr_matrix],
        List[scipy.sparse.csr_matrix],
        List[scipy.sparse.csr_matrix],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        scipy.sparse.csr_matrix,
    ],
    input_register: np.ndarray,
) -> scipy.sparse.csr_matrix:
    """Applies the Hamiltonian at a given time point on a state.

    Args:
        index_time (float): The index of the current time point
        operators_coefficients (Tuple[
            List[csr_matrix],
            List[csr_matrix],
            List[csr_matrix],
            ndarray,
            ndarray,
            ndarray,
            csr_matrix
        ]): A tuple containing the list of Rabi operators, the list of detuing operators,
        the list of local detuing operators, the list of Rabi frequencies, the list of global
        detuings, the list of local detunings and the interaction operator.
        input_register (ndarray): The input state which we apply the Hamiltonian to.
    Returns:
        (ndarray): The result
    """
    (
        rabi_ops,
        detuning_ops,
        local_detuning_ops,
        rabi_coefs,
        detuning_coefs,
        local_detuing_coefs,
        interaction_op,
    ) = operators_coefficients

    index_time = int(index_time)

    if len(rabi_coefs) > 0:
        # If there is driving field, the maximum of index_time is the maximum time index
        # for the driving field.
        # Note that, if there is more than one driving field, we assume that they have the
        # same number of coefficients
        max_index_time = len(rabi_coefs[0]) - 1
    else:
        # If there is no driving field, then the maxium of index_time is the maxium time
        # index for the shifting field.
        # Note that, if there is more than one shifting field, we assume that they have the
        # same number of coefficients
        # Note that, if there is no driving field nor shifting field, the initial state will
        # be returned, and the simulation would not reach here.
        max_index_time = len(local_detuing_coefs[0]) - 1

    # If the integrator uses intermediate time value that is larger than the maximum
    # time value specified, the final time value is used as an approximation.
    if index_time > max_index_time:
        index_time = max_index_time
        warnings.warn(
            "The solver uses intermediate time value that is "
            "larger than the maximum time value specified. "
            "The final time value of the specified range "
            "is used as an approximation."
        )

    # If the integrator uses intermediate time value that is larger than the minimum
    # time value specified, the final time value is used as an approximation.
    if index_time < 0:
        index_time = 0
        warnings.warn(
            "The solver uses intermediate time value that is "
            "smaller than the minimum time value specified. "
            "The first time value of the specified range "
            "is used as an approximation."
        )

    output_register = interaction_op.dot(input_register)

    # Add the driving fields
    for rabi_op, rabi_coef, detuning_op, detuning_coef in zip(
        rabi_ops, rabi_coefs, detuning_ops, detuning_coefs
    ):
        output_register += (rabi_coef[index_time] / 2) * rabi_op.dot(input_register)
        output_register += (np.conj(rabi_coef[index_time]) / 2) * rabi_op.H.dot(input_register)
        output_register -= detuning_coef[index_time] * detuning_op.dot(input_register)

    # Add the shifting fields
    for local_detuning_op, local_detuning_coef in zip(local_detuning_ops, local_detuing_coefs):
        output_register -= local_detuning_coef[index_time] * local_detuning_op.dot(input_register)

    return output_register


def _find_configuration_index(
    configuration: str,
) -> int:
    """Fining the decimal index corresponding to a configuration.

    Args:
        configuration (str): A configuration of atoms with each site being 'r' or 'g

    Returns:
        int: Corresponding decimal number. It is the index of the configuration
        in an array of all configurations sorted in ascending order accroding to their
        corresponding numerical values.
    """
    config_index = 0
    num_qubit = len(configuration)
    for idx in range(num_qubit):
        if configuration[-1 - idx] == "r":
            config_index += 2**idx
    return config_index


def apply_SPAM_noises(
    pre_sequence: List[int],
    state_dist: np.ndarray,
    configurations: List[str],
    noises: Dict[str, float] = {},
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], Union[float, int]]:
    """Post-processes the pre-sequence and state distribution to apply SPAM noise

    Args:
        pre_sequence (List[int]): The requested atom filling
        state_dist (np.ndarray): The state distribution corresponding to the
        evolved state. Must be normalized
        configurations (List[str]): The list of configurations that comply with the blockade
            approximation.
        noises (Dict[str, float], optional): A dictionary of noises and their parameters.
          Defaults to {}.

    Raises:
        ValueError: If the state distribution is not normalized

    Returns:
        Tuple[
            np.ndarray,
            np.ndarray,
            List[int],
            List[int],
            Union[float, int]
            ]: A tuple containing the post-processed pre-sequence distribution and
            state distribution, index of filled sites and empty sites and vacancy detection
            error rate.
    """
    pre_sequence = np.array(pre_sequence)

    non_empty_sites = np.nonzero(pre_sequence)[0]
    empty_sites = np.where(pre_sequence == 0)[0]

    num_qubits = len(non_empty_sites)
    num_vacancies = len(empty_sites)

    if (noise_type.ATOM_DETECTION in noises) or (noise_type.VACANCY_DETECTION in noises):
        eps_ad = noises.get(noise_type.ATOM_DETECTION, 0)
        eps_vd = noises.get(noise_type.VACANCY_DETECTION, 0)

        pre_seq_dist = np.zeros(num_qubits + num_vacancies)
        pre_seq_dist[non_empty_sites] = 1 - eps_ad
        pre_seq_dist[empty_sites] = eps_vd
    else:  # no detection noise, just return the pre-sequence
        eps_vd = 0
        pre_seq_dist = pre_sequence

    if not np.isclose(sum(state_dist), 1.0):
        raise ValueError("State distribution must be normalized!")

    full_space_size = 2**num_qubits  # size of the full Hilbert space
    noisy_state_dist = np.zeros(full_space_size)

    if full_space_size > len(configurations):  # blockade approximation in effect
        valid_idx = [_find_configuration_index(config) for config in configurations]
        noisy_state_dist[valid_idx] = state_dist
    else:
        noisy_state_dist = state_dist

    if (noise_type.GROUND_STATE_DETECTION in noises) or (
        noise_type.RYDBERG_STATE_DETECTION in noises
    ):
        # construct the confusion matrix to add noise to final distribution
        eps_g = noises.get(noise_type.GROUND_STATE_DETECTION, 0)
        eps_r = noises.get(noise_type.RYDBERG_STATE_DETECTION, 0)

        # For a single qubit, this matrix will multiply the probability vector
        # [P_ground_true ,P_rydberg_true], producing [P_ground_observed, P_rydberg_observed]
        # according to the conditional probabilities eps_g and eps_r.
        # The diagonal entries are the probability of correctly detecting each state.
        confusion_mtx_qubit = np.array([[1 - eps_g, eps_r], [eps_g, 1 - eps_r]])

        # We assume the detection of each qubit is independent from each other,
        # so the confusion matrix for the full system is simply a kronecker product
        # of copies of single-qubit confusion matrix.
        confusion_mtx_full = ft.reduce(np.kron, [confusion_mtx_qubit for _ in range(num_qubits)])

        # The dimension of noisy_state_dist is N = 2**n, the size of full Hilbert space
        # The dimension of confusion_mtx_full is N by N.
        # The dimension of noisy_state_dist is not changed by this transformation.
        noisy_state_dist = confusion_mtx_full @ noisy_state_dist  # convert probability
    # if no state detection noise, do not post-process the state distribution

    # need to return the vacancy detection error rate for subsequent sampling of
    # vacant sites with noise
    return (pre_seq_dist, noisy_state_dist, non_empty_sites, empty_sites, eps_vd)


def _get_lind_dict_T1(
    targets: Tuple[int], configurations: List[str]
) -> Dict[Tuple[int, int], float]:
    lind_T1 = {}  # The Lind term in the basis of configurations, as a dictionary

    # use dictionary to store index of configurations
    configuration_index = {config: ind for ind, config in enumerate(configurations)}

    # T1 decay terms
    for ind_1, config_1 in enumerate(configurations):
        for target in targets:
            # Converts a single atom from "r" to "g".
            if config_1[target] != "r":
                continue

            # Construct the state after applying the Lind operator
            bit_list = list(config_1)
            bit_list[target] = "g"
            config_2 = "".join(bit_list)

            # If the constructed state is in the Hilbert space,
            # add the corresponding matrix element to the Lind operator.
            if config_2 in configuration_index:
                lind_T1[(configuration_index[config_2], ind_1)] = 1

    return lind_T1


def _get_lind_dict_T2(
    targets: Tuple[int], configurations: List[str]
) -> Dict[Tuple[int, int], float]:
    lind_T2 = {}
    target = targets[0]  # only one target site for T2 decay

    for ind, config in enumerate(configurations):
        if config[target] == "r":
            lind_T2[(ind, ind)] = 1

    return lind_T2


def _get_ops_coefs_lind(
    targets: List[int], configurations: List[str], noises: dict
) -> Tuple[List[scipy.sparse.csr_matrix], np.ndarray]:
    if (noise_type.T_1 not in noises) and (noise_type.T_2 not in noises):
        warnings.warn(
            "No quantum channel noise speficied, using density matrix simulator"
            "is inefficient. Using the braket_ahs simulator is more efficient."
        )

    #  Get the lindblad operators as sparse matrices
    lind_ops, lind_coefs = [], []

    if noise_type.T_1 in noises:
        T1 = noises[noise_type.T_1]
        for target in targets:
            L_target_T1 = _get_sparse_from_dict(
                _get_lind_dict_T1((target,), configurations), len(configurations)
            )
            L_target_T1 = scipy.sparse.csr_matrix(L_target_T1, dtype=float)
            lind_ops.append(L_target_T1)
            lind_coefs.append(float(1 / T1))

    if noise_type.T_2 in noises:
        T2 = noises[noise_type.T_2]
        for target in targets:
            L_target_T2 = _get_sparse_from_dict(
                _get_lind_dict_T2((target,), configurations), len(configurations)
            )
            L_target_T2 = scipy.sparse.csr_matrix(L_target_T2, dtype=float)
            lind_ops.append(L_target_T2)
            lind_coefs.append(float(2 / T2))

    return (lind_ops, lind_coefs)


def _apply_lindbladian(
    index_time: int,
    h0_operators_coefficients: Tuple[
        List[scipy.sparse.csr_matrix],
        List[scipy.sparse.csr_matrix],
        List[scipy.sparse.csr_matrix],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        scipy.sparse.csr_matrix,
    ],
    lind_operators_coefficients: Tuple[
        List[scipy.sparse.csr_matrix],
        np.ndarray,
    ],
    input_register: np.ndarray,
) -> np.ndarray:
    hamiltonian = _get_hamiltonian(index_time, h0_operators_coefficients)

    hamiltonian_times_rho = hamiltonian @ input_register
    output_register = -1j * (hamiltonian_times_rho - hamiltonian_times_rho.conj().T)

    lind_ops, lind_coefs = lind_operators_coefficients

    # Add Lindblad terms
    for lind_op, lind_coef in zip(lind_ops, lind_coefs):
        decay_term = lind_op.conj().T @ lind_op @ input_register
        output_register += lind_coef * (
            lind_op @ input_register @ lind_op.conj().T - 0.5 * (decay_term + decay_term.conj().T)
        )

    return output_register
