import numpy as np
import pytest
from braket.ir.ahs.atom_arrangement import AtomArrangement

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    get_blockade_configurations,
    validate_config,
)

setup_1 = AtomArrangement(**{"sites": [[0, 0], [0, 3e-6], [0, 6e-6]], "filling": [1, 1, 1]})
atoms_coordinates_1 = np.array([[0, 0], [0, 3e-6], [0, 6e-6]])

config_1_1 = "grr"
config_1_2 = "ggg"
config_1_3 = "ggr"
config_1_4 = "grg"
config_1_5 = "rrr"
config_1_6 = "rgg"
config_1_7 = "rgr"
config_1_8 = "rrg"


@pytest.mark.parametrize(
    "para",
    (
        [config_1_1, atoms_coordinates_1, 0],
        [config_1_2, atoms_coordinates_1, 0],
        [config_1_3, atoms_coordinates_1, 0],
        [config_1_4, atoms_coordinates_1, 0],
        [config_1_5, atoms_coordinates_1, 0],
        [config_1_6, atoms_coordinates_1, 0],
        [config_1_7, atoms_coordinates_1, 0],
        [config_1_8, atoms_coordinates_1, 0],
        [config_1_2, atoms_coordinates_1, 3e-6],
        [config_1_3, atoms_coordinates_1, 3e-6],
        [config_1_4, atoms_coordinates_1, 3e-6],
        [config_1_6, atoms_coordinates_1, 3e-6],
        [config_1_7, atoms_coordinates_1, 3e-6],
    ),
)
def test_validate_config_1_pass(para):
    config, atoms_coordinates, blockade_radius = para[0], para[1], para[2]
    assert validate_config(config, atoms_coordinates, blockade_radius) is True


@pytest.mark.parametrize(
    "para",
    (
        [config_1_1, atoms_coordinates_1, 3e-6],
        [config_1_8, atoms_coordinates_1, 3e-6],
        [config_1_5, atoms_coordinates_1, 3e-6],
    ),
)
def test_validate_config_1_fail(para):
    config, atoms_coordinates, blockade_radius = para[0], para[1], para[2]
    assert validate_config(config, atoms_coordinates, blockade_radius) is False


@pytest.mark.parametrize("para", [[setup_1, 3e-6]])
def test_get_blockade_configurations_setup_1_blockade(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {"ggg": 0, "ggr": 1, "grg": 2, "rgg": 3, "rgr": 4}


@pytest.mark.parametrize("para", [[setup_1, 0]])
def test_get_blockade_configurations_setup_1_no_blockade(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {
        "ggg": 0,
        "ggr": 1,
        "grg": 2,
        "grr": 3,
        "rgg": 4,
        "rgr": 5,
        "rrg": 6,
        "rrr": 7,
    }


###
setup_2 = AtomArrangement(**{"sites": [[0, 0], [0, 3e-6], [0, 6e-6]], "filling": [1, 0, 1]})
atoms_coordinates_2 = np.array([[0, 0], [0, 6e-6]])

config_2_1 = "gg"
config_2_2 = "gr"
config_2_3 = "rg"
config_2_4 = "rr"


@pytest.mark.parametrize(
    "para",
    (
        [config_2_1, atoms_coordinates_2, 0],
        [config_2_2, atoms_coordinates_2, 0],
        [config_2_3, atoms_coordinates_2, 0],
        [config_2_4, atoms_coordinates_2, 0],
        [config_2_1, atoms_coordinates_2, 3e-6],
        [config_2_2, atoms_coordinates_2, 3e-6],
        [config_2_3, atoms_coordinates_2, 3e-6],
        [config_2_4, atoms_coordinates_2, 3e-6],
        [config_2_1, atoms_coordinates_2, 6e-6],
        [config_2_2, atoms_coordinates_2, 6e-6],
        [config_2_3, atoms_coordinates_2, 6e-6],
    ),
)
def test_validate_config_2_pass(para):
    config, atoms_coordinates, blockade_radius = para[0], para[1], para[2]
    assert validate_config(config, atoms_coordinates, blockade_radius) is True


@pytest.mark.parametrize("para", [[config_2_4, atoms_coordinates_2, 6e-6]])
def test_validate_config_2_fail(para):
    config, atoms_coordinates, blockade_radius = para[0], para[1], para[2]
    assert validate_config(config, atoms_coordinates, blockade_radius) is False


@pytest.mark.parametrize("para", [[setup_2, 3e-6]])
def test_get_blockade_configurations_setup_2_blockade_1(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {"gg": 0, "gr": 1, "rg": 2, "rr": 3}


@pytest.mark.parametrize("para", [[setup_2, 6e-6]])
def test_get_blockade_configurations_setup_2_blockade_2(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {"gg": 0, "gr": 1, "rg": 2}


@pytest.mark.parametrize("para", [[setup_2, 0]])
def test_get_blockade_configurations_setup_2_no_blockade(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {"gg": 0, "gr": 1, "rg": 2, "rr": 3}


###
setup_3 = AtomArrangement(**{"sites": [[0, 0], [0, 3e-6], [0, 6e-6]], "filling": [1, 1, 0]})
atoms_coordinates_3 = np.array([[0, 0], [0, 3e-6]])

config_3_1 = "gg"
config_3_2 = "gr"
config_3_3 = "rg"
config_3_4 = "rr"


@pytest.mark.parametrize(
    "para",
    (
        [config_3_1, atoms_coordinates_3, 0],
        [config_3_2, atoms_coordinates_3, 0],
        [config_3_3, atoms_coordinates_3, 0],
        [config_3_4, atoms_coordinates_3, 0],
        [config_3_1, atoms_coordinates_3, 3e-6],
        [config_3_2, atoms_coordinates_3, 3e-6],
        [config_3_3, atoms_coordinates_3, 3e-6],
        [config_3_1, atoms_coordinates_3, 6e-6],
        [config_3_2, atoms_coordinates_3, 6e-6],
        [config_3_3, atoms_coordinates_3, 6e-6],
    ),
)
def test_validate_config_3_pass(para):
    config, atoms_coordinates, blockade_radius = para[0], para[1], para[2]
    assert validate_config(config, atoms_coordinates, blockade_radius) is True


@pytest.mark.parametrize(
    "para", [[config_3_4, atoms_coordinates_3, 3e-6], [config_3_4, atoms_coordinates_3, 6e-6]]
)
def test_validate_config_3_fail(para):
    config, atoms_coordinates, blockade_radius = para[0], para[1], para[2]
    assert validate_config(config, atoms_coordinates, blockade_radius) is False


@pytest.mark.parametrize("para", [[setup_3, 3e-6], [setup_3, 6e-6]])
def test_get_blockade_configurations_setup_3_blockade(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {"gg": 0, "gr": 1, "rg": 2}


@pytest.mark.parametrize("para", [[setup_2, 0]])
def test_get_blockade_configurations_setup_3_no_blockade(para):
    lattice, blockade_radius = para[0], para[1]
    configurations = get_blockade_configurations(lattice, blockade_radius)
    assert configurations == {"gg": 0, "gr": 1, "rg": 2, "rr": 3}
