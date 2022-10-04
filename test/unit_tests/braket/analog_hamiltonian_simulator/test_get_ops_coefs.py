import numpy as np
import pytest
import scipy as sp
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import get_ops_coefs
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

a = 3
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF


eps = 1e-3

amplitude_1 = {"pattern": "uniform", "time_series": {"times": [0, 4e-6], "values": [10e6, 25e6]}}


detuning_1 = {
    "pattern": "uniform",
    "time_series": {"times": [0, 2e-6, 4e-6], "values": [-10e6, 25e6, 0]},
}

phase_1 = {
    "pattern": "uniform",
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [10, 20, -30, 40]},
}

shift_1 = {
    "pattern": [0.5, 1.0],
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [1e7, 2e7, -3e7, 4e7]},
}

setup1 = {"ahs_register": {"sites": [[0, 0], [0, 3e-6]], "filling": [1, 1]}}

program_1 = convert_unit(
    Program(
        setup=setup1,
        hamiltonian={
            "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
            "shiftingFields": [{"magnitude": shift_1}],
        },
    )
)

configurations_1 = ["gg", "gr", "rg", "rr"]


@pytest.mark.parametrize(
    "para", [[program_1, configurations_1, rydberg_interaction_coef, [1, 2, 3]]]
)
def test_get_ops_coefs(para):
    program, configurations, rydberg_interaction_coef, ts = para[0], para[1], para[2], para[3]
    (
        rabi_ops,
        detuning_ops,
        local_detuning_ops,
        rabi_coefs,
        detuning_coefs,
        local_detuing_coefs,
        interaction_op,
    ) = get_ops_coefs(program, configurations, rydberg_interaction_coef, ts)

    # Test the operator
    true_interaction_op = sp.sparse.csr_matrix(
        tuple([[rydberg_interaction_coef / (a**6)], [[3], [3]]]),
        shape=(len(configurations), len(configurations)),
    )

    true_detuning_op = sp.sparse.csr_matrix(
        tuple([[1, 1, 2], [[1, 2, 3], [1, 2, 3]]]), shape=(len(configurations), len(configurations))
    )

    true_rabi_ops = sp.sparse.csr_matrix(
        tuple([[1, 1, 1, 1], [[1, 2, 3, 3], [0, 0, 1, 2]]]),
        shape=(len(configurations), len(configurations)),
    )

    true_local_detuning_op = sp.sparse.csr_matrix(
        tuple([[1.0, 0.5, 1.5], [[1, 2, 3], [1, 2, 3]]]),
        shape=(len(configurations), len(configurations)),
    )

    assert (true_interaction_op != interaction_op).nnz == 0
    assert len(detuning_ops) == 1
    assert (true_detuning_op != detuning_ops[0]).nnz == 0
    assert len(rabi_ops) == 1
    assert (true_rabi_ops != rabi_ops[0]).nnz == 0
    assert len(local_detuning_ops) == 1
    assert (true_local_detuning_op != local_detuning_ops[0]).nnz == 0

    # Test the coefficients
    amplitude = program.hamiltonian.drivingFields[0].amplitude
    amplitude_times, amplitude_values = amplitude.time_series.times, amplitude.time_series.values
    phase = program.hamiltonian.drivingFields[0].phase
    phase_times, phase_values = phase.time_series.times, phase.time_series.values
    detuning = program.hamiltonian.drivingFields[0].detuning
    detuning_times, detuning_values = detuning.time_series.times, detuning.time_series.values
    shift = program.hamiltonian.shiftingFields[0].magnitude
    shift_times, shift_values = shift.time_series.times, shift.time_series.values

    true_rabi_coefs = []
    true_detuning_coefs = []
    true_local_detuing_coefs = []

    for t in ts:
        # figure out amplitude
        ind = np.searchsorted(amplitude_times, t, side="right") - 1
        amplitude_slope = (amplitude_values[ind + 1] - amplitude_values[ind]) / (
            amplitude_times[ind + 1] - amplitude_times[ind]
        )
        amplitude_t = amplitude_values[ind] + amplitude_slope * (t - amplitude_times[ind])

        # figure out phase
        ind = np.searchsorted(phase_times, t, side="right") - 1
        phase_t = phase_values[ind]

        # figure out detuning
        ind = np.searchsorted(detuning_times, t, side="right") - 1
        detuning_slope = (detuning_values[ind + 1] - detuning_values[ind]) / (
            detuning_times[ind + 1] - detuning_times[ind]
        )
        detuning_t = detuning_values[ind] + detuning_slope * (t - detuning_times[ind])

        # figure out shift
        for i in range(len(shift_times) - 1):
            if t < shift_times[i + 1]:
                ind = i
                break
        shift_t = shift_values[ind]
        shift_slope = (shift_values[ind + 1] - shift_values[ind]) / (
            shift_times[ind + 1] - shift_times[ind]
        )
        shift_t = shift_values[ind] + shift_slope * (t - shift_times[ind])

        true_rabi_coefs.append(float(amplitude_t) * np.exp(1j * float(phase_t)))
        true_detuning_coefs.append(float(detuning_t))
        true_local_detuing_coefs.append(float(shift_t))

    assert len(rabi_coefs) == 1
    assert len(detuning_coefs) == 1
    assert len(local_detuing_coefs) == 1

    assert all(
        [np.abs(item_1 - item_2) < eps for item_1, item_2 in zip(rabi_coefs[0], true_rabi_coefs)]
    )
    assert all(
        [
            np.abs(item_1 - item_2) < eps
            for item_1, item_2 in zip(detuning_coefs[0], true_detuning_coefs)
        ]
    )
    assert all(
        [
            np.abs(item_1 - item_2) < eps
            for item_1, item_2 in zip(local_detuing_coefs[0], true_local_detuing_coefs)
        ]
    )
