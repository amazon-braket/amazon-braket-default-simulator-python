from braket.ir.ahs.driving_field import DrivingField
from pydantic import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.field_validator_util import (
    validate_value_range_with_warning,
)


class DrivingFieldValidator(DrivingField):
    capabilities: CapabilitiesConstants

    # The last time point given in each of these `times` arrays must be equal.
    @root_validator(pre=True, skip_on_failure=True)
    def sequences_have_the_same_end_time(cls, values):
        fields = {"amplitude", "phase", "detuning"}
        end_times = []
        for field in fields:
            times = values[field]["sequence"]["times"]
            if times:
                end_times.append(values[field]["sequence"]["times"][-1])
        if end_times:
            if len(set(end_times)) != 1:
                raise ValueError(
                    f"The last timepoints for all the sequences are not equal. They are {end_times}"
                )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_pattern_is_uniform(cls, values):
        amplitude = values["amplitude"]
        if amplitude["pattern"] != "uniform":
            raise ValueError('Pattern of amplitude must be "uniform"')
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_values_within_range(cls, values):
        amplitude = values["amplitude"]
        capabilities = values["capabilities"]
        validate_value_range_with_warning(
            amplitude["sequence"]["values"],
            capabilities.GLOBAL_AMPLITUDE_VALUE_MIN,
            capabilities.GLOBAL_AMPLITUDE_VALUE_MAX,
            "amplitude",
        )
        return values

    # Validators for phase

    @root_validator(pre=True, skip_on_failure=True)
    def phase_pattern_is_uniform(cls, values):
        phase = values["phase"]
        if phase["pattern"] != "uniform":
            raise ValueError('Pattern of phase must be "uniform"')
        return values

    # Validators for detuning

    @root_validator(pre=True, skip_on_failure=True)
    def detuning_pattern_is_uniform(cls, values):
        detuning = values["detuning"]
        if detuning["pattern"] != "uniform":
            raise ValueError('Pattern of detuning must be "uniform"')
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def detuning_values_within_range(cls, values):
        detuning = values["detuning"]
        capabilities = values["capabilities"]
        validate_value_range_with_warning(
            detuning["sequence"]["values"],
            capabilities.GLOBAL_DETUNING_VALUE_MIN,
            capabilities.GLOBAL_DETUNING_VALUE_MAX,
            "detuning",
        )
        return values
