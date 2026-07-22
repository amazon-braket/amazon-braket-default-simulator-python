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

from pydantic import model_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.field_validator_util import (
    validate_value_range_with_warning,
)
from braket.ir.ahs.driving_field import DrivingField


class DrivingFieldValidator(DrivingField):
    capabilities: CapabilitiesConstants

    @model_validator(mode="before")
    @classmethod
    def validate_driving_field(cls, values):
        cls.sequences_have_the_same_end_time(values)
        cls.amplitude_pattern_is_uniform(values)
        cls.amplitude_values_within_range(values)
        cls.phase_pattern_is_uniform(values)
        cls.detuning_pattern_is_uniform(values)
        cls.detuning_values_within_range(values)
        return values

    # The last time point given in each of these `times` arrays must be equal.
    @staticmethod
    def sequences_have_the_same_end_time(values):
        fields = {"amplitude", "phase", "detuning"}
        end_times = []
        for field in fields:
            times = values[field]["time_series"]["times"]
            if times:
                end_times.append(values[field]["time_series"]["times"][-1])
        if end_times and len(set(end_times)) != 1:
            raise ValueError(
                f"The last timepoints for all the sequences are not equal. They are {end_times}"
            )

    @staticmethod
    def amplitude_pattern_is_uniform(values):
        amplitude = values["amplitude"]
        if amplitude["pattern"] != "uniform":
            raise ValueError('Pattern of amplitude must be "uniform"')

    @staticmethod
    def amplitude_values_within_range(values):
        amplitude = values["amplitude"]
        capabilities = values["capabilities"]
        validate_value_range_with_warning(
            amplitude["time_series"]["values"],
            capabilities.GLOBAL_AMPLITUDE_VALUE_MIN,
            capabilities.GLOBAL_AMPLITUDE_VALUE_MAX,
            "amplitude",
        )

    # Validators for phase

    @staticmethod
    def phase_pattern_is_uniform(values):
        phase = values["phase"]
        if phase["pattern"] != "uniform":
            raise ValueError('Pattern of phase must be "uniform"')

    # Validators for detuning

    @staticmethod
    def detuning_pattern_is_uniform(values):
        detuning = values["detuning"]
        if detuning["pattern"] != "uniform":
            raise ValueError('Pattern of detuning must be "uniform"')

    @staticmethod
    def detuning_values_within_range(values):
        detuning = values["detuning"]
        capabilities = values["capabilities"]
        validate_value_range_with_warning(
            detuning["time_series"]["values"],
            capabilities.GLOBAL_DETUNING_VALUE_MIN,
            capabilities.GLOBAL_DETUNING_VALUE_MAX,
            "detuning",
        )
