import warnings

from braket.ir.ahs.atom_array import AtomArray
from pydantic.class_validators import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)


class AtomArrayValidator(AtomArray):
    capabilities: CapabilitiesConstants

    # Each site has two coordinates (minItems=maxItems=2)
    @root_validator(pre=True, skip_on_failure=True)
    def sites_have_length_2(cls, values):
        sites = values["sites"]
        capabilities = values["capabilities"]
        for index, site in enumerate(sites):
            if len(site) != capabilities.DIMENSIONS:
                raise ValueError(
                    f"Site {index}({site}) has length {len(site)}; it must be "
                    f"{capabilities.DIMENSIONS}."
                )
        return values

    # All lattice sites should fit within a (BOUNDING_BOX_SIZE_X) x (BOUNDING_BOX_SIZE_Y)
    # bounding box. If not, a warning message will issue to remind the user that the SI
    # units are used here.
    @root_validator(pre=True, skip_on_failure=True)
    def sites_fit_in_bounding_box(cls, values):
        sites = values["sites"]
        if sites:
            capabilities = values["capabilities"]
            sorted_sites = sorted(sites, key=lambda xy: xy[0])
            biggest_x_distance = sorted_sites[-1][0] - sorted_sites[0][0]
            if biggest_x_distance > capabilities.BOUNDING_BOX_SIZE_X:
                warnings.warn(
                    f"Arrangement is too wide. Sites {sorted_sites[0]} and {sorted_sites[-1]} "
                    "have x-separation bigger than the typical scale "
                    f"({capabilities.BOUNDING_BOX_SIZE_X} meters). "
                    "The coordinates of the atoms should be specified in SI units."
                )

            if biggest_x_distance <= capabilities.BOUNDING_BOX_SIZE_X:
                sorted_sites = sorted(sites, key=lambda xy: xy[1])
                biggest_y_distance = sorted_sites[-1][1] - sorted_sites[0][1]
                if biggest_y_distance > capabilities.BOUNDING_BOX_SIZE_Y:
                    warnings.warn(
                        f"Arrangement is too tall. Sites {sorted_sites[0]} and {sorted_sites[-1]} "
                        "have y-separation bigger than the typical scale "
                        f"({capabilities.BOUNDING_BOX_SIZE_Y} meters). "
                        "The coordinates of the atoms should be specified in SI units."
                    )

        return values

    #  Filling has only integers which are either 0 or 1
    @root_validator(pre=True, skip_on_failure=True)
    def filling_contains_only_0_and_1(cls, values):
        filling = values["filling"]
        for idx, f in enumerate(filling):
            if f not in {0, 1}:
                raise ValueError(f"Invalid value at {idx} (value: {f}). Only 0 and 1 are allowed.")
        return values

    # Filling must have the same length as `lattice_sites`.
    @root_validator(pre=True, skip_on_failure=True)
    def filling_same_length_as_sites(cls, values):
        filling = values["filling"]
        expected_length = len(values["sites"])
        length = len(filling)
        if length != expected_length:
            raise ValueError(
                f"Filling length ({length}) does not match sites length ({expected_length})"
            )
        return values
