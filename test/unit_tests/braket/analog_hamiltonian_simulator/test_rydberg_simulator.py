import numpy as np
import pytest
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationTaskResult,
)

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator import RydbergAtomSimulator

device = RydbergAtomSimulator()


pi = np.pi

a = 3
Omega = 12 * pi
theta = 0.0  # 16.45
Delta1 = 20 * pi
Delta2 = 0.0  # 10*pi
duration = 4
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF

setup = {
    "atomArray": {
        "sites": [[0, 0], [0, a * 1e-6], [0, a * 2e-6], [0, a * 3e-6]],
        "filling": [1, 0, 1, 0],
    }
}
amplitude = {
    "pattern": "uniform",
    "sequence": {
        "times": [0, duration * 1e-06],
        "values": [Omega * 1e6, Omega * 1e6],
    },
}

phase = {
    "pattern": "uniform",
    "sequence": {
        "times": [0, duration * 1e-06],
        "values": [theta, theta],
    },
}

detuning = {
    "pattern": "uniform",
    "sequence": {
        "times": [0, duration * 1e-06],
        "values": [Delta1 * 1e6, Delta1 * 1e6],
    },
}

shift = {
    "sequence": {"times": [0, duration * 1e-06], "values": [Delta2 * 1e6, Delta2 * 1e6]},
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
    "program",
    [
        (program_full),
        (empty_program),
        (program_only_shiftingFields),
        (program_only_drivingFields),
    ],
)
def test_success_qutip_run(program):
    result = device.run(program, progress_bar=True, shots=100)
    assert isinstance(result, AnalogHamiltonianSimulationTaskResult)


@pytest.mark.parametrize(
    "program, error_message",
    [
        (program_full, "Shot = 0 is not implemented yet"),
        (empty_program, "Shot = 0 is not implemented yet"),
        (program_only_shiftingFields, "Shot = 0 is not implemented yet"),
        (program_only_drivingFields, "Shot = 0 is not implemented yet"),
    ],
)
def test_run_shot_0(program, error_message):
    with pytest.raises(NotImplementedError) as e:
        device.run(program, shots=0)
    assert error_message in str(e.value)
