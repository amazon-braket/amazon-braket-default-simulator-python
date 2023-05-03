import pytest
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _get_detuning_dict,
    _get_interaction_dict,
    _get_rabi_dict,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

a = 3
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF


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
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [1e7, 2e7, -3e7, 4e7]},
}

setup_1 = {"ahs_register": {"sites": [[0, 0], [0, a * 1e-6]], "filling": [1, 1]}}

program_1 = convert_unit(
    Program(
        setup=setup_1,
        hamiltonian={
            "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
            "shiftingFields": [{"magnitude": shift_1}],
        },
    )
)

configurations_1 = ["gg", "gr", "rg", "rr"]


@pytest.mark.parametrize("para", [[program_1, rydberg_interaction_coef, configurations_1]])
def test_get_interaction_dict_setup_1(para):
    program, rydberg_interaction_coef, configurations = para[0], para[1], para[2]
    interaction = _get_interaction_dict(program, rydberg_interaction_coef, configurations)

    assert interaction == dict({(3, 3): rydberg_interaction_coef / (a**6)})


def test_get_detuning_dict_configurations_1_1():
    assert _get_detuning_dict((0,), configurations_1) == dict({(2, 2): 1, (3, 3): 1})


def test_get_detuning_dict_configurations_1_2():
    assert _get_detuning_dict((1,), configurations_1) == dict({(1, 1): 1, (3, 3): 1})


def test_get_detuning_dict_configurations_1_3():
    assert _get_detuning_dict((0, 1), configurations_1) == dict({(1, 1): 1, (2, 2): 1, (3, 3): 2})


def test_get_rabi_dict_configurations_1_1():
    assert _get_rabi_dict((0,), configurations_1) == dict({(2, 0): 1, (3, 1): 1})


def test_get_rabi_dict_configurations_1_2():
    assert _get_rabi_dict((1,), configurations_1) == dict({(1, 0): 1, (3, 2): 1})


def test_get_rabi_dict_configurations_1_3():
    assert _get_rabi_dict((0, 1), configurations_1) == dict(
        {(1, 0): 1, (2, 0): 1, (3, 1): 1, (3, 2): 1}
    )


setup_2 = {"ahs_register": {"sites": [[0, 0], [0, a * 1e-6], [0, a * 2e-6]], "filling": [1, 0, 1]}}

program_2 = convert_unit(
    Program(
        setup=setup_2,
        hamiltonian={
            "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
            "shiftingFields": [{"magnitude": shift_1}],
        },
    )
)

configurations_2 = ["gg", "gr", "rg", "rr"]


@pytest.mark.parametrize("para", [[program_2, rydberg_interaction_coef, configurations_2]])
def test_get_interaction_dict_setup_2(para):
    program, rydberg_interaction_coef, configurations = para[0], para[1], para[2]
    interaction = _get_interaction_dict(program, rydberg_interaction_coef, configurations)

    assert interaction == dict({(3, 3): rydberg_interaction_coef / ((2 * a) ** 6)})


setup3 = {"ahs_register": {"sites": [[0, 0], [0, a * 1e-6], [0, a * 2e-6]], "filling": [1, 1, 1]}}

program_3 = convert_unit(
    Program(
        setup=setup3,
        hamiltonian={
            "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
            "shiftingFields": [{"magnitude": shift_1}],
        },
    )
)

configurations_3 = ["ggg", "ggr", "grg", "grr", "rgg", "rgr", "rrg", "rrr"]


@pytest.mark.parametrize("para", [[program_3, rydberg_interaction_coef, configurations_3]])
def test_get_interaction_dict_setup_3(para):
    program, rydberg_interaction_coef, configurations = para[0], para[1], para[2]
    interaction = _get_interaction_dict(program, rydberg_interaction_coef, configurations)

    assert pytest.approx(interaction) == dict(
        {
            (3, 3): rydberg_interaction_coef / (a**6),
            (5, 5): rydberg_interaction_coef / ((2 * a) ** 6),
            (6, 6): rydberg_interaction_coef / (a**6),
            (7, 7): 2 * rydberg_interaction_coef / (a**6)
            + rydberg_interaction_coef / ((2 * a) ** 6),
        }
    )


def test_get_rabi_dict_configurations_3_1():
    assert _get_rabi_dict((0,), configurations_3) == dict(
        {(4, 0): 1, (5, 1): 1, (6, 2): 1, (7, 3): 1}
    )


def test_get_rabi_dict_configurations_3_2():
    assert _get_rabi_dict((1,), configurations_3) == dict(
        {(2, 0): 1, (3, 1): 1, (6, 4): 1, (7, 5): 1}
    )


def test_get_rabi_dict_configurations_3_3():
    assert _get_rabi_dict((2,), configurations_3) == dict(
        {(1, 0): 1, (3, 2): 1, (5, 4): 1, (7, 6): 1}
    )


def test_get_rabi_dict_configurations_3_4():
    assert _get_rabi_dict((0, 1), configurations_3) == dict(
        {(4, 0): 1, (5, 1): 1, (6, 2): 1, (7, 3): 1, (2, 0): 1, (3, 1): 1, (6, 4): 1, (7, 5): 1}
    )


def test_get_rabi_dict_configurations_3_5():
    assert _get_rabi_dict((0, 2), configurations_3) == dict(
        {(1, 0): 1, (3, 2): 1, (5, 4): 1, (7, 6): 1, (4, 0): 1, (5, 1): 1, (6, 2): 1, (7, 3): 1}
    )


def test_get_rabi_dict_configurations_3_6():
    assert _get_rabi_dict((1, 2), configurations_3) == dict(
        {(2, 0): 1, (3, 1): 1, (6, 4): 1, (7, 5): 1, (1, 0): 1, (3, 2): 1, (5, 4): 1, (7, 6): 1}
    )


def test_get_detuning_dict_configurations_3_1():
    assert _get_detuning_dict((0,), configurations_3) == dict(
        {(4, 4): 1, (5, 5): 1, (6, 6): 1, (7, 7): 1}
    )


def test_get_detuning_dict_configurations_3_2():
    assert _get_detuning_dict((1,), configurations_3) == dict(
        {(2, 2): 1, (3, 3): 1, (6, 6): 1, (7, 7): 1}
    )


def test_get_detuning_dict_configurations_3_3():
    assert _get_detuning_dict((2,), configurations_3) == dict(
        {(1, 1): 1, (3, 3): 1, (5, 5): 1, (7, 7): 1}
    )


def test_get_detuning_dict_configurations_3_4():
    assert _get_detuning_dict((0, 1), configurations_3) == dict(
        {(2, 2): 1, (3, 3): 1, (6, 6): 2, (7, 7): 2, (4, 4): 1, (5, 5): 1}
    )


def test_get_detuning_dict_configurations_3_5():
    assert _get_detuning_dict((0, 2), configurations_3) == dict(
        {(1, 1): 1, (3, 3): 1, (5, 5): 2, (7, 7): 2, (4, 4): 1, (6, 6): 1}
    )


def test_get_detuning_dict_configurations_3_6():
    assert _get_detuning_dict((0, 1, 2), configurations_3) == dict(
        {(1, 1): 1, (2, 2): 1, (3, 3): 2, (6, 6): 2, (7, 7): 3, (4, 4): 1, (5, 5): 2}
    )


configurations_4 = ["gg", "gr", "rg"]


@pytest.mark.parametrize("para", [[program_1, rydberg_interaction_coef, configurations_4]])
def test_get_interaction_dict_setup_4(para):
    program, rydberg_interaction_coef, configurations = para[0], para[1], para[2]
    interaction = _get_interaction_dict(program, rydberg_interaction_coef, configurations)

    assert interaction == dict()


def test_get_detuning_dict_configurations_4_1():
    assert _get_detuning_dict((0,), configurations_4) == dict({(2, 2): 1})


def test_get_detuning_dict_configurations_4_2():
    assert _get_detuning_dict((1,), configurations_4) == dict({(1, 1): 1})


def test_get_detuning_dict_configurations_4_3():
    assert _get_detuning_dict((0, 1), configurations_4) == dict({(1, 1): 1, (2, 2): 1})


def test_get_rabi_dict_configurations_4_1():
    assert _get_rabi_dict((0,), configurations_4) == dict({(2, 0): 1})


def test_get_rabi_dict_configurations_4_2():
    assert _get_rabi_dict((1,), configurations_4) == dict({(1, 0): 1})


def test_get_rabi_dict_configurations_4_3():
    assert _get_rabi_dict((0, 1), configurations_4) == dict({(1, 0): 1, (2, 0): 1})
