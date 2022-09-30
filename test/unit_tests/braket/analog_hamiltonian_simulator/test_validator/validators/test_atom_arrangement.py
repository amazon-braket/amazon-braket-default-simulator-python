import pytest
from braket.ir.ahs.atom_arrangement import AtomArrangement
from pydantic.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.atom_arrangement import (
    AtomArrangementValidator,
)


@pytest.fixture
def atom_array_data():
    return {
        "sites": [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
        "filling": [1, 0, 1, 0],
    }


@pytest.fixture
def mock_atom_array_data():
    data = {
        "sites": [],
        "filling": [1, 1, 1, 1],
    }
    return AtomArrangement.parse_obj(data).dict()


def test_valid_atom_array(atom_array_data, device_capabilities_constants):
    try:
        AtomArrangementValidator(capabilities=device_capabilities_constants, **atom_array_data)
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "sites, warning_message",
    [
        (
            [[-50.1e-6, 50.0e-6], [0.0, 0.0], [0.0, 0.000001], [50.0e-6, -50.0e-6]],
            "Arrangement is too wide. Sites [-5.01e-05, 5e-05] and [5e-05, -5e-05] have "
            "x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [[-50.0e-6, 50.1e-6], [0.0, 0.0], [0.0, 0.000001], [50.0e-6, -50.0e-6]],
            "Arrangement is too wide. Sites [-5e-05, 5.01e-05] and [5e-05, -5e-05] have "
            "x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [
                [-22.7e-6, 28.3e-6],
                [31.3e-6, 31.3e-6],
                [31.3e-6, -68.8e-6],
                [22.7e-6, 28.3e-6],
            ],
            "Arrangement is too tall. Sites [3.13e-05, -6.88e-05] and [3.13e-05, 3.13e-05] "
            "have y-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [[50.0e-6, 100.1e-6], [100.1e-6, 50.0e-6], [50.0e-6, 0.0], [0.0, 50.0e-6]],
            "Arrangement is too wide. Sites [0.0, 5e-05] and [0.0001001, 5e-05] have "
            "x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [[0.0, 0.0], [0.0, 0.000001], [6.71e-5, -3.9e-5], [-3.3e-5, 6.11e-5]],
            "Arrangement is too wide. Sites [-3.3e-05, 6.11e-05] and [6.71e-05, -3.9e-05] "
            "have x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
    ],
)
# Rule: The lattice sites must define an area with maximum dimensions of
#       lattice.area.width, lattice.area.height
def test_atom_array_sites_fit_in_bounding_box(
    sites, warning_message, mock_atom_array_data, device_capabilities_constants
):
    mock_atom_array_data["sites"] = sites
    _assert_warning_is_produced_for_atom_array(
        mock_atom_array_data, warning_message, device_capabilities_constants
    )


@pytest.mark.parametrize(
    "sites, filling, error_message",
    [
        (
            [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
            [1, 0, 1, 0, 1],
            "Filling length (5) does not match sites length (4)",
        ),
        (
            [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
            [1, 0, 1],
            "Filling length (3) does not match sites length (4)",
        ),
    ],
)
# Rule: Lattice filling must specify occupancy/vacancy for each lattice site
def test_atom_array_filling_same_length_as_sites(
    sites, filling, error_message, mock_atom_array_data, device_capabilities_constants
):
    mock_atom_array_data["sites"] = sites
    mock_atom_array_data["filling"] = filling
    _assert_validation_error_is_raised_for_atom_array(
        mock_atom_array_data, error_message, device_capabilities_constants
    )


def test_atom_array_sites_have_length_2(mock_atom_array_data, device_capabilities_constants):
    sites = [[0, 0, 0], [0, 4e-6, 5e-6]]
    mock_atom_array_data["sites"] = sites
    error_message = "Site 0([0, 0, 0]) has length 3; it must be 2."
    _assert_validation_error_is_raised_for_atom_array(
        mock_atom_array_data, error_message, device_capabilities_constants
    )


def test_atom_array_filling_contains_only_0_and_1(
    mock_atom_array_data, device_capabilities_constants
):
    filling = [0, 1, 2, 1]
    mock_atom_array_data["filling"] = filling
    error_message = "Invalid value at 2 (value: 2). Only 0 and 1 are allowed."
    _assert_validation_error_is_raised_for_atom_array(
        mock_atom_array_data, error_message, device_capabilities_constants
    )


def _assert_validation_error_is_raised_for_atom_array(
    data, error_message, device_capabilities_constants
):
    with pytest.raises(ValidationError) as e:
        AtomArrangementValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)


def _assert_warning_is_produced_for_atom_array(
    data, warning_message, device_capabilities_constants
):
    with pytest.warns(UserWarning) as e:
        AtomArrangementValidator(capabilities=device_capabilities_constants, **data)
    # print(e[-1].message)
    assert warning_message in str(e[-1].message)
