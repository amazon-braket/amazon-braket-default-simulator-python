from decimal import Decimal
from typing import Tuple

from pydantic.v1.class_validators import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.atom_arrangement import (
    AtomArrangementValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators import (
    DeviceCapabilitiesConstants,
)


def _y_distance(site_1: Tuple[Decimal, Decimal], site_2: Tuple[Decimal, Decimal]) -> Decimal:
    # Compute the y-separation between two sets of 2-D points, (x1, y1) and (x2, y2)

    return Decimal(abs(site_1[1] - site_2[1]))


class DeviceAtomArrangementValidator(AtomArrangementValidator):
    capabilities: DeviceCapabilitiesConstants

    @root_validator(pre=True, skip_on_failure=True)
    def sites_not_empty(cls, values):
        sites = values["sites"]
        if not sites:
            raise ValueError("Sites can not be empty.")
        return values

    # The maximum allowable precision in the coordinates is SITE_PRECISION
    @root_validator(pre=True, skip_on_failure=True)
    def sites_defined_with_right_precision(cls, values):
        sites = values["sites"]
        capabilities = values["capabilities"]
        for idx, s in enumerate(sites):
            if not all(
                [Decimal(str(coordinate)) % capabilities.SITE_PRECISION == 0 for coordinate in s]
            ):
                raise ValueError(
                    f"Coordinates {idx}({s}) is defined with too high precision;\
                        they must be multiples of {capabilities.SITE_PRECISION} meters"
                )
        return values

    # Number of sites must not exceeds MAX_SITES
    @root_validator(pre=True, skip_on_failure=True)
    def sites_not_too_many(cls, values):
        sites = values["sites"]
        capabilities = values["capabilities"]
        num_sites = len(sites)
        if num_sites > capabilities.MAX_SITES:
            raise ValueError(
                f"There are too many sites ({num_sites}); there must be at most\
                    {capabilities.MAX_SITES} sites"
            )
        return values

    # The y coordinates of any two lattice sites must either be equal
    # or differ by at least MIN_ROW_DISTANCE.
    @root_validator(pre=True, skip_on_failure=True)
    def sites_in_rows(cls, values):
        sites = values["sites"]
        capabilities = values["capabilities"]
        sorted_sites = sorted(sites, key=lambda xy: xy[1])
        min_allowed_distance = capabilities.MIN_ROW_DISTANCE
        if capabilities.LOCAL_RYDBERG_CAPABILITIES:
            min_allowed_distance = Decimal("0.000002")
        for s1, s2 in zip(sorted_sites[:-1], sorted_sites[1:]):
            row_distance = _y_distance(s1, s2)
            if row_distance == 0:
                continue
            if row_distance < min_allowed_distance:
                raise ValueError(
                    f"Sites {s1} and site {s2} have y-separation ({row_distance}). It must\
                        either be exactly zero or not smaller than {min_allowed_distance} meters"
                )
        return values

    # The number of filled lattice sites must not exceed MAX_FILLED_SITES.
    @root_validator(pre=True, skip_on_failure=True)
    def atom_number_limit(cls, values):
        filling = values["filling"]
        capabilities = values["capabilities"]
        qubits = sum(filling)
        if qubits > capabilities.MAX_FILLED_SITES:
            raise ValueError(
                f"Filling has {qubits} '1' entries; is must have not\
                    more than {capabilities.MAX_FILLED_SITES}"
            )
        return values
