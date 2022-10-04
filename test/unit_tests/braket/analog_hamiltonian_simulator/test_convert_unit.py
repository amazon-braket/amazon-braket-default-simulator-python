import pytest
from braket.ir.ahs.physical_field import PhysicalField
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import FIELD_UNIT, SPACE_UNIT, TIME_UNIT
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
    convert_unit_for_field,
)

amplitude1 = {"pattern": "uniform", "time_series": {"times": [0, 4e-6], "values": [10e6, 25e6]}}


detuning1 = {
    "pattern": "uniform",
    "time_series": {"times": [0, 2e-6, 4e-6], "values": [-10e6, 25e6, 0]},
}

phase1 = {
    "pattern": "uniform",
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [10, 20, -30, 40]},
}

shift1 = {
    "pattern": [0.0, 1.0, 0.5, 0.0],
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [10, 20, -30, 40]},
}

setup1 = {"ahs_register": {"sites": [[0, 0]], "filling": [1]}}

program1 = Program(
    setup=setup1,
    hamiltonian={
        "drivingFields": [{"amplitude": amplitude1, "phase": phase1, "detuning": detuning1}],
        "shiftingFields": [{"magnitude": shift1}],
    },
)


@pytest.mark.parametrize("field", [amplitude1, detuning1, shift1])
def test_convert_unit_for_field_amp_det(field):
    newfield = PhysicalField(**convert_unit_for_field(PhysicalField(**field), True))
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


@pytest.mark.parametrize("field", [phase1])
def test_convert_unit_for_field_phase(field):
    newfield = PhysicalField(**convert_unit_for_field(PhysicalField(**field), False))
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


@pytest.mark.parametrize("program", [program1])
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
