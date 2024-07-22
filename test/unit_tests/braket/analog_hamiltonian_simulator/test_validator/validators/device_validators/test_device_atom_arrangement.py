from decimal import Decimal

import pytest
from pydantic.v1.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.atom_arrangement import AtomArrangement
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators import (
    DeviceAtomArrangementValidator,
)


@pytest.fixture
def atom_arrangement_data():
    return {
        "sites": [[0, 0], [0, 5e-6], [5e-6, 0], [5e-6, 10e-6]],
        "filling": [1, 0, 1, 0],
    }


@pytest.fixture
def mock_atom_arrangement_data():
    data = {
        "sites": [],
        "filling": [],
    }
    return AtomArrangement.parse_obj(data).dict()


@pytest.mark.parametrize(
    "sites, filling, error_message",
    [
        (
            [],
            [],
            "Sites can not be empty.",
        ),
        (
            [],
            [1, 0, 1, 0],
            "Filling length (4) does not match sites length (0)",
        ),
        (
            [
                [Decimal("0.0"), Decimal("0.0")],
                [Decimal("0.0"), Decimal("4e-6")],
                [Decimal("5e-6"), Decimal("0.0")],
                [Decimal("5e-6"), Decimal("4e-6")],
            ],
            [],
            "Filling length (0) does not match sites length (4)",
        ),
    ],
)
def test_atom_arrangement_sites_or_fillings_empty(
    sites, filling, error_message, mock_atom_arrangement_data, capabilities_with_local_rydberg
):
    mock_atom_arrangement_data["sites"] = sites
    mock_atom_arrangement_data["filling"] = filling
    _assert_atom_arrangement(
        mock_atom_arrangement_data, error_message, capabilities_with_local_rydberg
    )


def test_valid_atom_array(atom_arrangement_data, capabilities_with_local_rydberg):
    try:
        DeviceAtomArrangementValidator(
            capabilities=capabilities_with_local_rydberg, **atom_arrangement_data
        )
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "sites, filling ,error_message",
    [
        (
            [[1.1e-6, 1.1e-6], [2.2e-6, 7.31e-6]],
            [1, 0],
            "Coordinates 1([2.2e-06, 7.31e-06]) is defined with too high precision;"
            "they must be multiples of 1E-7 meters",
        ),
        (
            [[3.201e-6, 0.0], [0.00, 4.1e-6], [-1.101e-6, 8.11e-6]],
            [1, 0, 1],
            "Coordinates 0([3.201e-06, 0.0]) is defined with too high precision;"
            "they must be multiples of 1E-7 meters",
        ),
    ],
)
# Rule: Lattice coordinates have a maximum precision of lattice.geometry.position_resolution
def test_atom_arrangement_sites_defined_with_right_precision(
    sites, filling, error_message, mock_atom_arrangement_data, non_local_capabilities_constants
):
    mock_atom_arrangement_data["sites"] = sites
    mock_atom_arrangement_data["filling"] = filling
    _assert_atom_arrangement(
        mock_atom_arrangement_data, error_message, non_local_capabilities_constants
    )


def _assert_atom_arrangement(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        DeviceAtomArrangementValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)


def test_atom_arrangement_sites_not_too_many(
    mock_atom_arrangement_data, non_local_capabilities_constants
):
    sites = [[0.0, 0.0]] * (non_local_capabilities_constants.MAX_SITES + 1)
    filling = [0] * (non_local_capabilities_constants.MAX_SITES + 1)
    mock_atom_arrangement_data["sites"] = sites
    mock_atom_arrangement_data["filling"] = filling
    error_message = "There are too many sites (9); there must be at most 8 sites"
    _assert_atom_arrangement(
        mock_atom_arrangement_data, error_message, non_local_capabilities_constants
    )


@pytest.mark.parametrize(
    "sites, filling,error_message",
    [
        (
            [[Decimal("0.0"), Decimal("0.0")], [Decimal("5e-6"), Decimal("2.4e-6")]],
            [0, 1],
            "Sites [Decimal('0.0'), Decimal('0.0')] and site [Decimal('0.000005'), "
            "Decimal('0.0000024')] have y-separation (0.0000024). It must either be "
            "exactly zero or not smaller than 0.000004 meters",
        ),
        (
            [[Decimal("0.0"), Decimal("0.0")], [Decimal("5e-6"), Decimal("1.4e-6")]],
            [0, 1],
            "Sites [Decimal('0.0'), Decimal('0.0')] and site [Decimal('0.000005'), "
            "Decimal('0.0000014')] have y-separation (0.0000014). It must either "
            "be exactly zero or not smaller than 0.000004 meters",
        ),
    ],
)
# Rule: All sites in the lattice must be separated vertically
# by at least lattice.geometry.spacing_vertical_min
def test_atom_arrangement_sites_in_rows(
    sites,
    filling,
    error_message,
    mock_atom_arrangement_data,
    non_local_capabilities_constants,
):
    mock_atom_arrangement_data["sites"] = sites
    mock_atom_arrangement_data["filling"] = filling
    _assert_atom_arrangement(
        mock_atom_arrangement_data,
        error_message,
        non_local_capabilities_constants,
    )


def test_atom_arrangement_filling_atom_number_limit(
    mock_atom_arrangement_data, non_local_capabilities_constants
):
    filling = [1] * (non_local_capabilities_constants.MAX_FILLED_SITES + 1)
    mock_atom_arrangement_data["filling"] = filling
    mock_atom_arrangement_data["sites"] = [
        [0, 0],
        [0, 0.00004],
        [0, 0.00005],
        [0.00005, 0.00004],
        [0.00005, 0],
    ]
    error_message = "Filling has 5 '1' entries; it must have not more than 4"
    _assert_atom_arrangement(
        mock_atom_arrangement_data, error_message, non_local_capabilities_constants
    )
