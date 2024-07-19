import pytest
from pydantic.v1.error_wrappers import ValidationError
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators.\
    device_driving_field import DeviceDrivingFieldValidator
from braket.analog_hamiltonian_simulator.rydberg.validators.driving_field \
    import DrivingField


    
@pytest.fixture
def mock_driving_field_data():
    data = {
        "amplitude": {
            "pattern": "uniform",
            "time_series": {
                "times": [],
                "values": [],
            },
        },
        "phase": {
            "pattern": "uniform",
            "time_series": {
                "times": [],
                "values": [],
            },
        },
        "detuning": {
            "pattern": "uniform",
            "time_series": {
                "times": [],
                "values": [],
            },
        },
    }
    return DrivingField.parse_obj(data).dict()


@pytest.mark.parametrize(
    "times, field_name, error_message",
    [
        (
            [0.0, 12.1e-9],
            "amplitude",
            "time point 1 (1.21E-8) of amplitude time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9],
            "phase",
            "time point 1 (1.21E-8) of phase time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9],
            "detuning",
            "time point 1 (1.21E-8) of detuning time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 4e-6],
            "amplitude",
            "time point 1 (1.21E-8) of amplitude time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 4e-6],
            "phase",
            "time point 1 (1.21E-8) of phase time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 4e-6],
            "detuning",
            "time point 1 (1.21E-8) of detuning time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 22.1e-9],
            "amplitude",
            "time point 1 (1.21E-8) of amplitude time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 22.1e-9],
            "phase",
            "time point 1 (1.21E-8) of phase time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 12.1e-9, 22.1e-9],
            "detuning",
            "time point 1 (1.21E-8) of detuning time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 22.1e-9, 12.1e-9],
            "amplitude",
            "time point 1 (2.21E-8) of amplitude time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 22.1e-9, 12.1e-9],
            "phase",
            "time point 1 (2.21E-8) of phase time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
        (
            [0.0, 22.1e-9, 12.1e-9],
            "detuning",
            "time point 1 (2.21E-8) of detuning time_series is defined with too many digits; it must be an integer multple of 1E-9",
        ),
    ],
)
# Rule: The times for any component of the effective Hamiltonian have a maximum precision of rydberg.global.time_resolution
def test_driving_field_time_precision_is_correct(
    times, field_name, error_message, mock_driving_field_data, non_local_capabilities_constants
):
    mock_driving_field_data[field_name]["time_series"]["times"] = times
    _assert_driving_field(mock_driving_field_data, error_message, non_local_capabilities_constants)

@pytest.mark.parametrize(
    "times, field_name, error_message",
    [
        (
            [0.0, 9e-9, 25e-9],
            "amplitude",
            "Time points of amplitude time_series, 0 (0.0) and 1 (9e-09), are too close; they are separated by 9e-09 seconds. It must be at least 1E-8 seconds",
        ),
        (
            [0.0, 9e-9, 25e-9],
            "phase",
            "Time points of phase time_series, 0 (0.0) and 1 (9e-09), are too close; they are separated by 9e-09 seconds. It must be at least 1E-8 seconds",
        ),
        (
            [0.0, 9e-9, 25e-9],
            "detuning",
            "Time points of detuning time_series, 0 (0.0) and 1 (9e-09), are too close; they are separated by 9e-09 seconds. It must be at least 1E-8 seconds",
        ),
        (
            [0.0, 9e-9, 25e-9, 30e-9],
            "amplitude",
            "Time points of amplitude time_series, 0 (0.0) and 1 (9e-09), are too close; they are separated by 9e-09 seconds. It must be at least 1E-8 seconds",
        ),
        (
            [0.0, 9e-9, 25e-9, 30e-9],
            "phase",
            "Time points of phase time_series, 0 (0.0) and 1 (9e-09), are too close; they are separated by 9e-09 seconds. It must be at least 1E-8 seconds",
        ),
        (
            [0.0, 9e-9, 25e-9, 30e-9],
            "detuning",
            "Time points of detuning time_series, 0 (0.0) and 1 (9e-09), are too close; they are separated by 9e-09 seconds. It must be at least 1E-8 seconds",
        ),
    ],
)
# Rule: The times for any component of the effective Hamiltonian must be spaced by at least rydberg.global.time_delta_min
def test_driving_field_timepoint_not_too_close(
    times, field_name, error_message, mock_driving_field_data, non_local_capabilities_constants
):
    mock_driving_field_data[field_name]["time_series"]["times"] = times
    _assert_driving_field(mock_driving_field_data, error_message, non_local_capabilities_constants)


@pytest.mark.parametrize(
    "values, field_name, error_message",
    [
        (
            [2.5e7, 2.5e7, 2.5e7, 0.0],
            "amplitude",
            "The values of the Rabi frequency at the first and last time points are 25000000.0, 0.0; they both must be both 0.",
        ),
        (
            [0.0, 2.5e7, 2.5e7, 2.5e7],
            "amplitude",
            "The values of the Rabi frequency at the first and last time points are 0.0, 25000000.0; they both must be both 0.",
        ),
        (
            [2.5e7, 2.5e7, 2.5e7, 2.5e7],
            "amplitude",
            "The values of the Rabi frequency at the first and last time points are 25000000.0, 25000000.0; they both must be both 0.",
        ),
    ],
)
#  Rule: The global Rydberg Rabi frequency amplitude must start and end at 0.
def test_driving_field_amplitude_start_and_end_values(
    values, field_name, error_message, mock_driving_field_data, non_local_capabilities_constants
):
    mock_driving_field_data[field_name]["time_series"]["values"] = values
    _assert_driving_field(mock_driving_field_data, error_message, non_local_capabilities_constants)
    

@pytest.mark.parametrize(
    "values, times, field_name, error_message",
    [
        (
            [0.0, 2.5e7, 0.0],
            [0.0, 0.01e-6, 2.0e-6],
            "amplitude",
            "For the amplitude field, rate of change of values (between the 0-th and the 1-th times) is 2500000000000000.0, more than 250000000000000",
        ),
        (
            [0.0, 2.5e7, 2.5e7, 0.0],
            [0.0, 0.01e-6, 3.2e-6, 3.21e-6],
            "amplitude",
            "For the amplitude field, rate of change of values (between the 0-th and the 1-th times) is 2500000000000000.0, more than 250000000000000",
        ), 
        (
            [0.0, 2.5e7, 2.5e7, 0.0],
            [0.0, 0.01e-6, 3.2e-6, 3.21e-6],
            "detuning",
            "For the detuning field, rate of change of values (between the 0-th and the 1-th times) is 2500000000000000.0, more than 250000000000000",
        ),
    ]
)
def test_driving_field_slopes_not_too_steep(
    values, times, field_name, error_message, mock_driving_field_data, non_local_capabilities_constants
):
    mock_driving_field_data[field_name]["time_series"]["values"] = values
    mock_driving_field_data[field_name]["time_series"]["times"] = times
    _assert_driving_field(mock_driving_field_data, error_message, non_local_capabilities_constants)
    
def test_phase_values_start_with_0(mock_driving_field_data, non_local_capabilities_constants):
    mock_driving_field_data["phase"]["time_series"]["values"] = [0.1, 2.5e7]
    error_message = "The first value of of driving field phase is 0.1; it must be 0."
    _assert_driving_field(mock_driving_field_data, error_message, non_local_capabilities_constants)
    
def _assert_driving_field(data, error_message, device_capabilities_constants):
    with pytest.raises(ValidationError) as e:
        DeviceDrivingFieldValidator(capabilities=device_capabilities_constants, **data)
    assert error_message in str(e.value)