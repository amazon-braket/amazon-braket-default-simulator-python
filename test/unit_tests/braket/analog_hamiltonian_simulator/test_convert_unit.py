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

import pytest
from braket.ir.ahs.physical_field import PhysicalField
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import FIELD_UNIT, SPACE_UNIT, TIME_UNIT
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    _convert_unit_for_field,
    convert_unit,
)

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
    "pattern": [0.0, 1.0, 0.5, 0.0],
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [10, 20, -30, 40]},
}

setup_1 = {"ahs_register": {"sites": [[0, 0]], "filling": [1]}}

program_1 = Program(
    setup=setup_1,
    hamiltonian={
        "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
        "shiftingFields": [{"magnitude": shift_1}],
    },
)


@pytest.mark.parametrize("field", [amplitude_1, detuning_1, shift_1])
def test__convert_unit_for_field_amp_det(field):
    newfield = PhysicalField(**_convert_unit_for_field(PhysicalField(**field), True))
    truth = PhysicalField(
        **{
            "pattern": field["pattern"],
            "time_series": {
                "times": [float(time) / TIME_UNIT for time in field["time_series"]["times"]],
                "values": [float(value) / FIELD_UNIT for value in field["time_series"]["values"]],
            },
        }
    )
    assert newfield == truth


@pytest.mark.parametrize("field", [phase_1])
def test__convert_unit_for_field_phase(field):
    newfield = PhysicalField(**_convert_unit_for_field(PhysicalField(**field), False))
    truth = PhysicalField(
        **{
            "pattern": field["pattern"],
            "time_series": {
                "times": [float(time) / TIME_UNIT for time in field["time_series"]["times"]],
                "values": [float(value) for value in field["time_series"]["values"]],
            },
        }
    )

    assert newfield == truth


@pytest.mark.parametrize("program", [program_1])
def test_convert_unit(program):
    newprogram = convert_unit(program)

    truth = Program(
        setup={
            "ahs_register": {
                "sites": [
                    [float(site[0]) / SPACE_UNIT, float(site[1]) / SPACE_UNIT]
                    for site in program.setup.ahs_register.sites
                ],
                "filling": program.setup.ahs_register.filling,
            }
        },
        hamiltonian={
            "drivingFields": [
                {
                    "amplitude": {
                        "pattern": program.hamiltonian.drivingFields[0].amplitude.pattern,
                        "time_series": {
                            "times": [
                                float(time) / TIME_UNIT
                                for time in program.hamiltonian.drivingFields[
                                    0
                                ].amplitude.time_series.times
                            ],
                            "values": [
                                float(value) / FIELD_UNIT
                                for value in program.hamiltonian.drivingFields[
                                    0
                                ].amplitude.time_series.values
                            ],
                        },
                    },
                    "detuning": {
                        "pattern": program.hamiltonian.drivingFields[0].detuning.pattern,
                        "time_series": {
                            "times": [
                                float(time) / TIME_UNIT
                                for time in program.hamiltonian.drivingFields[
                                    0
                                ].detuning.time_series.times
                            ],
                            "values": [
                                float(value) / FIELD_UNIT
                                for value in program.hamiltonian.drivingFields[
                                    0
                                ].detuning.time_series.values
                            ],
                        },
                    },
                    "phase": {
                        "pattern": program.hamiltonian.drivingFields[0].phase.pattern,
                        "time_series": {
                            "times": [
                                float(time) / TIME_UNIT
                                for time in program.hamiltonian.drivingFields[
                                    0
                                ].phase.time_series.times
                            ],
                            "values": [
                                float(value)
                                for value in program.hamiltonian.drivingFields[
                                    0
                                ].phase.time_series.values
                            ],
                        },
                    },
                }
            ],
            "shiftingFields": [
                {
                    "magnitude": {
                        "pattern": program.hamiltonian.shiftingFields[0].magnitude.pattern,
                        "time_series": {
                            "times": [
                                float(time) / TIME_UNIT
                                for time in program.hamiltonian.shiftingFields[
                                    0
                                ].magnitude.time_series.times
                            ],
                            "values": [
                                float(value) / FIELD_UNIT
                                for value in program.hamiltonian.shiftingFields[
                                    0
                                ].magnitude.time_series.values
                            ],
                        },
                    }
                }
            ],
        },
    )
    assert newprogram == truth
