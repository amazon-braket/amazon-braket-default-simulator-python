import numpy as np
import pytest
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationTaskResult,
)

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator import RydbergAtomSimulator
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

device = RydbergAtomSimulator()


pi = np.pi

a = 3
rabi_frequency = 12 * pi
rabi_phase = 0.0
detuning_1 = 20 * pi
detuning_2 = 0.0
duration = 4
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF

setup = {
    "ahs_register": {
        "sites": [[0, 0], [0, a * 1e-6], [0, a * 2e-6], [0, a * 3e-6]],
        "filling": [1, 0, 1, 0],
    }
}
amplitude = {
    "pattern": "uniform",
    "time_series": {
        "times": [0, duration * 1e-06],
        "values": [rabi_frequency * 1e6, rabi_frequency * 1e6],
    },
}

phase = {
    "pattern": "uniform",
    "time_series": {
        "times": [0, duration * 1e-06],
        "values": [rabi_phase, rabi_phase],
    },
}

detuning = {
    "pattern": "uniform",
    "time_series": {
        "times": [0, duration * 1e-06],
        "values": [detuning_1 * 1e6, detuning_1 * 1e6],
    },
}

shift = {
    "time_series": {"times": [0, duration * 1e-06], "values": [detuning_2 * 1e6, detuning_2 * 1e6]},
    "pattern": [0.0, 1.0, 0.5, 0.0],
}


program_full = Program(
    setup=setup,
    hamiltonian={
        "drivingFields": [{"amplitude": amplitude, "phase": phase, "detuning": detuning}],
        "shiftingFields": [{"magnitude": shift}],
    },
)

empty_program = Program(
    setup=setup,
    hamiltonian={
        "drivingFields": [],
        "shiftingFields": [],
    },
)

program_only_shiftingFields = Program(
    setup=setup,
    hamiltonian={
        "drivingFields": [],
        "shiftingFields": [{"magnitude": shift}],
    },
)

program_only_drivingFields = Program(
    setup=setup,
    hamiltonian={
        "drivingFields": [{"amplitude": amplitude, "phase": phase, "detuning": detuning}],
        "shiftingFields": [],
    },
)


def test_device_properties():
    properties = device.properties
    assert properties.action == {"braket.ir.ahs.program": {}}


def test_device_initialization():
    device.initialize_simulation()


@pytest.mark.parametrize(
    "program, rydberg_interaction_coef, blockade_radius",
    [
        (program_full, rydberg_interaction_coef, a * 1e-6),
        (empty_program, rydberg_interaction_coef, a * 1e-6),
        (program_only_shiftingFields, rydberg_interaction_coef, a * 1e-6),
        (program_only_drivingFields, rydberg_interaction_coef, a * 1e-6),
    ],
)
def test_success_run(program, rydberg_interaction_coef, blockade_radius):
    result = device.run(
        program,
        rydberg_interaction_coef=rydberg_interaction_coef,
        blockade_radius=blockade_radius,
        shots=100,
    )
    assert isinstance(result, AnalogHamiltonianSimulationTaskResult)


@pytest.mark.parametrize(
    "program",
    [
        (program_full),
        (empty_program),
        (program_only_shiftingFields),
        (program_only_drivingFields),
    ],
)
def test_success_run_without_args(program):
    result = device.run(program, shots=100)
    assert isinstance(result, AnalogHamiltonianSimulationTaskResult)


@pytest.mark.parametrize(
    "program, error_message",
    [
        (program_full, "Shot = 0 is not currently implemented"),
        (empty_program, "Shot = 0 is not currently implemented"),
        (program_only_shiftingFields, "Shot = 0 is not currently implemented"),
        (program_only_drivingFields, "Shot = 0 is not currently implemented"),
    ],
)
def test_run_shot_0(program, error_message):
    with pytest.raises(NotImplementedError) as e:
        device.run(program, shots=0)
    assert error_message in str(e.value)


zero_field = {
    "pattern": "uniform",
    "time_series": {
        "times": [0, 1e-8],
        "values": [0, 0],
    },
}


zero_program = convert_unit(
    Program(
        setup={
            "ahs_register": {
                "sites": [[0, i * a] for i in range(11)],
                "filling": [1 for _ in range(11)],
            }
        },
        hamiltonian={
            "drivingFields": [
                {"amplitude": zero_field, "phase": zero_field, "detuning": zero_field}
            ],
            "shiftingFields": [],
        },
    )
)


def test_scipy_run_for_large_system():
    result = device.run(zero_program, steps=1)
    assert isinstance(result, AnalogHamiltonianSimulationTaskResult)
