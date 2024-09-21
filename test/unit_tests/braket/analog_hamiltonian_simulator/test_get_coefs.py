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

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _get_coefs,
    _interpolate_time_series,
)
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

setup_1 = {"ahs_register": {"sites": [[0, 0], [0, 3e-6]], "filling": [1, 1]}}

program_1 = convert_unit(
    Program(
        setup=setup_1,
        hamiltonian={
            "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
            "localDetuning": [{"magnitude": shift_1}],
        },
    )
)

configurations1 = ["gg", "gr", "rg", "rr"]


@pytest.mark.parametrize(
    "para",
    [
        [
            [1e-6, 2e-6, 3e-6],
            amplitude_1["time_series"]["times"],
            amplitude_1["time_series"]["values"],
        ],
        [
            [1e-6, 2e-6, 3e-6],
            detuning_1["time_series"]["times"],
            detuning_1["time_series"]["values"],
        ],
        [[1e-6, 2e-6, 3e-6], phase_1["time_series"]["times"], phase_1["time_series"]["values"]],
        [[1e-6, 2e-6, 3e-6], shift_1["time_series"]["times"], shift_1["time_series"]["values"]],
    ],
)
def test_get_func(para):
    ts, times, values = para
    vals = [_interpolate_time_series(t, times, values) for t in ts]

    trueval = []
    for t in ts:
        for i in range(len(times) - 1):
            if t < times[i + 1]:
                ind = i
                break
        trueval.append(
            values[ind]
            + (values[ind + 1] - values[ind]) / (times[ind + 1] - times[ind]) * (t - times[ind])
        )

    assert np.allclose(vals, trueval)


@pytest.mark.parametrize(
    "method, error_message",
    [
        ("square_wave", "`method` can only be `piecewise_linear` or `piecewise_constant`."),
    ],
)
def test_interpolate_time_series_error_message(method, error_message):
    with pytest.raises(ValueError) as e:
        _interpolate_time_series(0, [0, 1], [0, 1], method)
    assert error_message in str(e.value)


@pytest.mark.parametrize("para", [[[1, 2, 3], program_1]])
def test_get_coefs(para):
    ts, program = para[0], para[1]
    rabi_coefs, detuning_coefs, local_detuing_coefs = _get_coefs(program, ts)

    amplitude = program.hamiltonian.drivingFields[0].amplitude
    amplitude_times, amplitude_values = amplitude.time_series.times, amplitude.time_series.values
    phase = program.hamiltonian.drivingFields[0].phase
    phase_times, phase_values = phase.time_series.times, phase.time_series.values
    detuning = program.hamiltonian.drivingFields[0].detuning
    detuning_times, detuning_values = detuning.time_series.times, detuning.time_series.values
    shift = program.hamiltonian.localDetuning[0].magnitude
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
        ind = np.searchsorted(shift_times, t, side="right") - 1
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
