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
from decimal import Decimal
from typing import List

import numpy as np
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)


def validate_value_range_with_warning(
    values: list[Decimal], min_value: Decimal, max_value: Decimal, name: str
) -> None:
    """
    Validate the given list of values against the allowed range

    Args:
        values (list[Decimal]): The given list of values to be validated
        min_value (Decimal): The minimal value allowed
        max_value (Decimal): The maximal value allowed
        name (str): The name of the field corresponds to the values
    """
    # Raise ValueError if at any item in the values is outside the allowed range
    # [min_value, max_value]
    for i, value in enumerate(values):
        if not min_value <= value <= max_value:
            warnings.warn(
                f"Value {i} ({value}) in {name} time series outside the typical range "
                f"[{min_value}, {max_value}]. The values should  be specified in SI units."
            )
            break  # Only one warning messasge will be issued


def validate_net_detuning_with_warning(
    program: Program,
    time_points: np.ndarray,
    global_detuning_coefs: np.ndarray,
    local_detuning_patterns: List,
    local_detuning_coefs: np.ndarray,
    capabilities: CapabilitiesConstants,
) -> Program:
    """
    Validate the given program for the net detuning of all the atoms at all time points

    Args:
        program (Program): The given program
        time_points (np.ndarray): The time points for both global and local detunings
        global_detuning_coefs (np.ndarray): The values of global detuning
        local_detuning_patterns (List): The pattern of local detuning
        local_detuning_coefs (np.ndarray): The values of local detuning
        capabilities (CapabilitiesConstants): The capability constants

    Returns:
        program (Program): The given program
    """

    for time_ind, time in enumerate(time_points):

        # Get the contributions from all the global detunings
        # (there could be multiple global driving fields) at the time point
        values_global_detuning = sum(
            [detuning_coef[time_ind] for detuning_coef in global_detuning_coefs]
        )

        for atom_index in range(len(local_detuning_patterns[0])):
            # Get the contributions from local detuning at the time point
            values_local_detuning = sum(
                [
                    shift_coef[time_ind] * float(detuning_pattern[atom_index])
                    for detuning_pattern, shift_coef in zip(
                        local_detuning_patterns, local_detuning_coefs
                    )
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

                # Return immediately if there is an atom has net detuning
                # exceeding MAX_NET_DETUNING at a time point
                return program


def validate_time_separation(times: List[Decimal], min_time_separation: Decimal, name: str) -> None:
    """
    Used in Device Emulation; Validate that the time points in a time series are separated by at 
    least min_time_separation. 
    
    Args:
        times (List[Decimal]): A list of time points in a time series. 
        min_time_separation (Decimal): The minimal amount of time any two time points should be 
            separated by. 
        name (str): The name of the time series, used for logging.
    
    Raises: 
        ValueError: If any two subsequent time points (assuming the time points are sorted
        in ascending order) are separated by less than min_time_separation.
    """
    for i in range(len(times) - 1):
        time_diff = times[i + 1] - times[i]
        if time_diff < min_time_separation:
            raise ValueError(
                f"Time points of {name} time_series, {i} ({times[i]}) and "
                f"{i + 1} ({times[i + 1]}), are too close; they are separated "
                f"by {time_diff} seconds. It must be at least {min_time_separation} seconds"
            )

def validate_value_precision(values: List[Decimal], max_precision: Decimal, name: str) -> None:
    """
    Used in Device Emulation; Validate that the precision of a set of values do not
    exceed max_precision.
    
    Args:
        times (List[Decimal]): A list of values from a time series to validate. 
        max_precision (Decimal): The maximum allowed precision. 
        name (str): The name of the time series, used for logging.
    
    Raises: 
        ValueError: If any of the given values is defined with precision exceeding max_precision.
    """
    for idx, v in enumerate(values):
        if v % max_precision != 0:
            raise ValueError(
                f"Value {idx} ({v}) in {name} time_series is defined with too many digits; "
                f"it must be an integer multiple of {max_precision}"
            )

def validate_max_absolute_slope(
    times: List[Decimal], values: List[Decimal], max_slope: Decimal, name: str
):
    """
    Used in Device Emulation; Validate that the magnitude of the slope between any 
    two subsequent points in a time series (time points provided in ascending order) does not
    exceed max_slope.
    
    Args:
        times (List[Decimal]): A list of time points in a time series. 
        max_slope (Decimal): The maximum allowed rate of change between points in the time series.
        name (str): The name of the time series, used for logging.
    
    Raises: 
        ValueError: if at any time the time series (times, values)
        rises/falls faster than allowed.
    """
    for idx in range(len(values) - 1):
        slope = (values[idx + 1] - values[idx]) / (times[idx + 1] - times[idx])
        if abs(slope) > max_slope:
            raise ValueError(
                f"For the {name} field, rate of change of values "
                f"(between the {idx}-th and the {idx + 1}-th times) "
                f"is {abs(slope)}, more than {max_slope}"
            )

def validate_time_precision(times: List[Decimal], time_precision: Decimal, name: str):
    """
    Used in Device Emulation; Validate that the precision of a set of time points do not
    exceed max_precision.
    
    Args:
        times (List[Decimal]): A list of time points to validate. 
        max_precision (Decimal): The maximum allowed precision. 
        name (str): The name of the time series, used for logging.
    
    Raises: 
        ValueError: If any of the given time points is defined with
        precision exceeding max_precision.
    """
    for idx, t in enumerate(times):
        if t % time_precision != 0:
            raise ValueError(
                f"time point {idx} ({t}) of {name} time_series is "
                f"defined with too many digits; it must be an "
                f"integer multiple of {time_precision}"
            )
