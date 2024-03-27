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
from braket.ir.ahs.atom_arrangement import AtomArrangement
from pydantic.v1.error_wrappers import ValidationError

from braket.analog_hamiltonian_simulator.rydberg.validators.atom_arrangement import (
    AtomArrangementValidator,
)


@pytest.fixture
def atom_arrangement_data():
    return {
        "sites": [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
        "filling": [1, 0, 1, 0],
    }


@pytest.fixture
def mock_atom_arrangement_data():
    data = {
        "sites": [],
        "filling": [1, 1, 1, 1],
    }
    return AtomArrangement.parse_obj(data).dict()


def test_valid_atom_array(atom_arrangement_data, device_capabilities_constants):
    try:
        AtomArrangementValidator(
            capabilities=device_capabilities_constants, **atom_arrangement_data
        )
    except ValidationError as e:
        pytest.fail(f"Validate test is failing : {str(e)}")


@pytest.mark.parametrize(
    "sites, warning_message",
    [
        (
            [[-50.1e-6, 50.0e-6], [0.0, 0.0], [0.0, 5e-6], [50.0e-6, -50.0e-6]],
            "Sites [-5.01e-05, 5e-05] and [5e-05, -5e-05] have "
            "x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [[-50.0e-6, 50.1e-6], [0.0, 0.0], [0.0, 5e-6], [50.0e-6, -50.0e-6]],
            "Sites [-5e-05, 5.01e-05] and [5e-05, -5e-05] have "
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
            "Sites [3.13e-05, -6.88e-05] and [3.13e-05, 3.13e-05] "
            "have y-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [[50.0e-6, 100.1e-6], [100.1e-6, 50.0e-6], [50.0e-6, 0.0], [0.0, 50.0e-6]],
            "Sites [0.0, 5e-05] and [0.0001001, 5e-05] have "
            "x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
        (
            [[0.0, 0.0], [0.0, 5e-6], [6.71e-5, -3.9e-5], [-3.3e-5, 6.11e-5]],
            "Sites [-3.3e-05, 6.11e-05] and [6.71e-05, -3.9e-05] "
            "have x-separation bigger than the typical scale (0.0001 meters). "
            "The coordinates of the atoms should be specified in SI units.",
        ),
    ],
)
# Rule: The lattice sites must define an area with maximum dimensions of
#       lattice.area.width, lattice.area.height
def test_atom_array_sites_fit_in_bounding_box(
    sites, warning_message, mock_atom_arrangement_data, device_capabilities_constants
):
    mock_atom_arrangement_data["sites"] = sites
    _assert_warning_is_produced_for_atom_array(
        mock_atom_arrangement_data, warning_message, device_capabilities_constants
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
    sites, filling, error_message, mock_atom_arrangement_data, device_capabilities_constants
):
    mock_atom_arrangement_data["sites"] = sites
    mock_atom_arrangement_data["filling"] = filling
    _assert_validation_error_is_raised_for_atom_array(
        mock_atom_arrangement_data, error_message, device_capabilities_constants
    )


def test_atom_array_sites_have_length_2(mock_atom_arrangement_data, device_capabilities_constants):
    sites = [[0, 0, 0], [0, 4e-6, 5e-6]]
    mock_atom_arrangement_data["sites"] = sites
    error_message = "Site 0([0, 0, 0]) has length 3; it must be 2."
    _assert_validation_error_is_raised_for_atom_array(
        mock_atom_arrangement_data, error_message, device_capabilities_constants
    )


def test_atom_array_filling_contains_only_0_and_1(
    mock_atom_arrangement_data, device_capabilities_constants
):
    filling = [0, 1, 2, 1]
    mock_atom_arrangement_data["filling"] = filling
    error_message = "Invalid value at 2 (value: 2). Only 0 and 1 are allowed."
    _assert_validation_error_is_raised_for_atom_array(
        mock_atom_arrangement_data, error_message, device_capabilities_constants
    )


@pytest.mark.parametrize(
    "sites, warning_message",
    [
        (
            [[0.0, 0.0], [2.4e-6, 0.0], [8.8e-6, 0.0], [0.0, 4.4e-6]],
            (
                "Sites 0([0.0, 0.0]) and site 1([2.4e-06, 0.0]) are too close. "
                "Their Euclidean distance (0.0000024 meters) is smaller than "
                "the typical scale (0.000004 meters). "
                "The coordinates of the sites should be specified in SI units."
            ),
        ),
        (
            [[0.0, 0.0], [0.0, 4.0e-6], [2.6e-6, 4.0e-6], [2.6e-6, 9.0e-6]],
            (
                "Sites 1([0.0, 4e-06]) and site 2([2.6e-06, 4e-06]) are too close. "
                "Their Euclidean distance (0.0000026 meters) is smaller than "
                "the typical scale (0.000004 meters). "
                "The coordinates of the sites should be specified in SI units."
            ),
        ),
    ],
)
# Rule: All sites in the lattice must be at least lattice.geometry.spacing_radial_min
# from each other
def test_atom_arrangement_sites_not_too_close(
    sites, warning_message, mock_atom_arrangement_data, device_capabilities_constants
):
    mock_atom_arrangement_data["sites"] = sites
    _assert_warning_is_produced_for_atom_array(
        mock_atom_arrangement_data, warning_message, device_capabilities_constants
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
    assert warning_message in str(e[-1].message)
