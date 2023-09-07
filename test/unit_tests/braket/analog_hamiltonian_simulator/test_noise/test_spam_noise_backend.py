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

a = 4
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

noise_ad = {"atom_detection": 0.18}
noise_grd_ryd = {"ground_state_detection": 0.35, "rydberg_state_detection": 0.45}
noise_all = {
    "atom_detection": 0.23,
    "ground_state_detection": 0.25,
    "rydberg_state_detection": 0.27,
}

shots = 100


@pytest.mark.parametrize(
    "noises",
    [
        (noise_ad),
        (noise_grd_ryd),
        (noise_all),
    ],
)
def test_noisy_backend(noises):
    result = device.run(program_full, shots=shots, noises=noises)
    assert isinstance(result, AnalogHamiltonianSimulationTaskResult)
