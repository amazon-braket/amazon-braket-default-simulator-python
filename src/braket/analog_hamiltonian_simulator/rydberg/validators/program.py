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

import warnings
from copy import deepcopy

import numpy as np
from braket.ir.ahs.program_v1 import Program
from pydantic.v1 import root_validator

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import _get_coefs
from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)


class ProgramValidator(Program):
    capabilities: CapabilitiesConstants

    # The pattern of the shifting field must have the same length as the lattice_sites
    @root_validator(pre=True, skip_on_failure=True)
    def local_detuning_pattern_has_the_same_length_as_atom_array_sites(cls, values):
        num_sites = len(values["setup"]["ahs_register"]["sites"])
        for idx, local_detuning in enumerate(
            values["hamiltonian"]["localDetuning"]
            if "localDetuning" in values["hamiltonian"].keys()
            else values["hamiltonian"]["localDetuning"]
        ):
            pattern_size = len(local_detuning["magnitude"]["pattern"])
            if num_sites != pattern_size:
                raise ValueError(
                    f"The length of pattern ({pattern_size}) of local detuning {idx} must equal "
                    f"the number of atom array sites ({num_sites})."
                )
        return values

    # If there is local detuning, the net value of detuning for each atom
    # should not exceed a max detuning value
    @root_validator(pre=True, skip_on_failure=True)
    def net_detuning_must_not_exceed_max_net_detuning(cls, values):
        capabilities = values["capabilities"]  # device_capabilities

        # Extract the program and the fields
        program = deepcopy(values)
        del program["capabilities"]
        program = Program.parse_obj(program)
        driving_fields = program.hamiltonian.drivingFields
        local_detuning = program.hamiltonian.localDetuning

        # If no local detuning, we simply return the values
        # because there are separate validators to validate
        # the global driving fields in the program
        if not len(local_detuning):
            return values

        detuning_times = [
            local_detune.magnitude.time_series.times for local_detune in local_detuning
        ]

        # Merge the time points for different shifting terms and detuning term
        all_times = set(sum(detuning_times, []))
        for driving_field in driving_fields:
            all_times.update(driving_field.detuning.time_series.times)
        time_points = sorted(all_times)

        # Get the time-dependent functions for the detuning and shifts
        _, detuning_coefs, shift_coefs = _get_coefs(program, time_points)

        # Get the detuning pattern
        detuning_patterns = [local_detune.magnitude.pattern for local_detune in local_detuning]

        # For each time point, check that each atom has net detuning less than the threshold
        for time_ind, time in enumerate(time_points):

            # Get the contributions from all the global detunings
            # (there could be multiple global driving fields) at the time point
            values_global_detuning = sum(
                [detuning_coef[time_ind] for detuning_coef in detuning_coefs]
            )

            for atom_index in range(len(detuning_patterns[0])):
                # Get the contributions from local detuning at the time point
                values_local_detuning = sum(
                    [
                        shift_coef[time_ind] * float(detuning_pattern[atom_index])
                        for detuning_pattern, shift_coef in zip(detuning_patterns, shift_coefs)
                    ]
                )

                # The net detuning is the sum of both the global and local detunings
                detuning_to_check = np.real(values_local_detuning + values_global_detuning)

                # Issue a warning if the absolute value of the net detuning is
                # beyond MAX_NET_DETUNING
                if abs(detuning_to_check) > capabilities.MAX_NET_DETUNING:
                    warnings.warn(
                        f"Atom {atom_index} has net detuning {detuning_to_check} rad/s "
                        f"at time {time} seconds, which is outside the typical range "
                        f"[{-capabilities.MAX_NET_DETUNING}, {capabilities.MAX_NET_DETUNING}]."
                        f"Numerical instabilities may occur during simulation."
                    )
                    return values

        return values
