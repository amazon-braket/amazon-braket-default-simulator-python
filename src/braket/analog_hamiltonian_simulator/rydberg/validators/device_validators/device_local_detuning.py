from pydantic.v1.class_validators import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.device_capabilities_constants import (
    DeviceCapabilitiesConstants,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.field_validator_util import (
    validate_max_absolute_slope,
    validate_time_precision,
    validate_time_separation,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.local_detuning import (
    LocalDetuningValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.physical_field import PhysicalField


class DeviceLocalDetuningValidator(LocalDetuningValidator):
    capabilities: DeviceCapabilitiesConstants

    @root_validator(pre=True, skip_on_failure=True)
    def check_local_rydberg_capabilities(cls, values):
        capabilities = values["capabilities"]
        if not capabilities.LOCAL_RYDBERG_CAPABILITIES:
            raise ValueError(
                "Local Rydberg capabilities information has not been "
                "provided for local detuning."
            )
        return values

    # Rule: The number of locally-addressed sites must not exceed
    # rydberg.local.number_local_detuning_sites
    @root_validator(pre=True, skip_on_failure=True)
    def magnitude_pattern_have_not_too_many_nonzeros(cls, values):
        magnitude = values["magnitude"]
        capabilities = values["capabilities"]
        pattern = magnitude["pattern"]
        num_nonzeros = sum([p != 0.0 for p in pattern])
        if num_nonzeros > capabilities.LOCAL_MAX_NONZERO_PATTERN_VALUES:
            raise ValueError(
                f"Number of nonzero magnitude pattern values is {num_nonzeros}; "
                f"it must not be more than {capabilities.LOCAL_MAX_NONZERO_PATTERN_VALUES}"
            )
        return values

    # Rule: The Rydberg local detuning times have a
    # resolution of at least rydberg.local.time_resolution
    @root_validator(pre=True, skip_on_failure=True)
    def magnitude_time_precision_is_correct(cls, values):
        magnitude = values["magnitude"]
        capabilities = values["capabilities"]
        magnitude_obj = PhysicalField.parse_obj(magnitude)
        validate_time_precision(
            magnitude_obj.time_series.times, capabilities.LOCAL_TIME_PRECISION, "magnitude"
        )
        return values

    # Rule: The Rydberg local detuning times must be spaced at least rydberg.local.time_delta_min
    @root_validator(pre=True, skip_on_failure=True)
    def magnitude_timepoint_not_too_close(cls, values):
        magnitude = values["magnitude"]
        capabilities = values["capabilities"]
        times = magnitude["time_series"]["times"]
        validate_time_separation(times, capabilities.LOCAL_MIN_TIME_SEPARATION, "magnitude")
        return values

    # Rule: The Rydberg local detuning slew rate must
    # not exceed rydberg.local.detuning_slew_rate_max
    @root_validator(pre=True, skip_on_failure=True)
    def magnitude_slopes_not_too_steep(cls, values):
        magnitude = values["magnitude"]
        capabilities = values["capabilities"]
        magnitude_times = magnitude["time_series"]["times"]
        magnitude_values = magnitude["time_series"]["values"]
        if magnitude_times and magnitude_values:
            validate_max_absolute_slope(
                magnitude_times,
                magnitude_values,
                capabilities.LOCAL_MAGNITUDE_SLOPE_MAX,
                "magnitude",
            )
        return values

    # Rule: The Rydberg local detuning must start and end at 0.
    @root_validator(pre=True, skip_on_failure=True)
    def local_detuning_start_and_end_values(cls, values):
        magnitude = values["magnitude"]
        time_series = magnitude["time_series"]
        time_series_values = time_series["values"]
        if time_series_values:
            start_value, end_value = time_series_values[0], time_series_values[-1]
            if start_value != 0 or end_value != 0:
                raise ValueError(
                    f"The values of the shifting field magnitude time series at the first "
                    f"and last time points are {start_value}, {end_value}; "
                    f"they both must be nonzero."
                )
        return values
