from decimal import Decimal

import pytest

from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators import (
    DeviceCapabilitiesConstants,
)


@pytest.fixture
def non_local_capabilities_constants():
    capabilities_dict = {
        "BOUNDING_BOX_SIZE_X": Decimal("0.000075"),
        "BOUNDING_BOX_SIZE_Y": Decimal("0.000076"),
        "DIMENSIONS": 2,
        "GLOBAL_AMPLITUDE_SLOPE_MAX": Decimal("250000000000000"),
        "GLOBAL_AMPLITUDE_VALUE_MAX": Decimal("15800000.0"),
        "GLOBAL_AMPLITUDE_VALUE_MIN": Decimal("0.0"),
        "GLOBAL_AMPLITUDE_VALUE_PRECISION": Decimal("400.0"),
        "GLOBAL_DETUNING_SLOPE_MAX": Decimal("250000000000000"),
        "GLOBAL_DETUNING_VALUE_MAX": Decimal("125000000.0"),
        "GLOBAL_DETUNING_VALUE_MIN": Decimal("-125000000.0"),
        "GLOBAL_DETUNING_VALUE_PRECISION": Decimal("0.2"),
        "GLOBAL_MIN_TIME_SEPARATION": Decimal("1E-8"),
        "GLOBAL_PHASE_VALUE_MAX": Decimal("99.0"),
        "GLOBAL_PHASE_VALUE_MIN": Decimal("-99.0"),
        "GLOBAL_PHASE_VALUE_PRECISION": Decimal("5E-7"),
        "GLOBAL_TIME_PRECISION": Decimal("1E-9"),
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX": None,
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN": None,
        "LOCAL_MAGNITUDE_SLOPE_MAX": None,
        "LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES": None,
        "LOCAL_MIN_TIME_SEPARATION": None,
        "LOCAL_RYDBERG_CAPABILITIES": False,
        "LOCAL_TIME_PRECISION": None,
        "MAGNITUDE_PATTERN_VALUE_MAX": None,
        "MAGNITUDE_PATTERN_VALUE_MIN": None,
        "MAX_FILLED_SITES": 4,
        "MAX_NET_DETUNING": None,
        "MAX_SITES": 8,
        "MAX_TIME": Decimal("0.000004"),
        "MIN_DISTANCE": Decimal("0.000004"),
        "MIN_ROW_DISTANCE": Decimal("0.000004"),
        "SITE_PRECISION": Decimal("1E-7"),
    }
    return DeviceCapabilitiesConstants(**capabilities_dict)


@pytest.fixture
def capabilities_with_local_rydberg(non_local_capabilities_constants):
    local_rydberg_constants = {
        "LOCAL_RYDBERG_CAPABILITIES": True,
        "LOCAL_TIME_PRECISION": Decimal("1e-9"),
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX": Decimal("125000000.0"),
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN": Decimal("0.0"),
        "LOCAL_MAGNITUDE_SLOPE_MAX": Decimal("1256600000000000.0"),
        "LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES": Decimal("5e-06"),
        "LOCAL_MIN_TIME_SEPARATION": Decimal("1E-8"),
        "LOCAL_MAX_NONZERO_PATTERN_VALUES": 200,
        "MAGNITUDE_PATTERN_VALUE_MAX": Decimal("1.0"),
        "MAGNITUDE_PATTERN_VALUE_MIN": Decimal("0.0"),
    }
    capabilities_dict = non_local_capabilities_constants.dict()
    capabilities_dict.update(local_rydberg_constants)
    return DeviceCapabilitiesConstants(**capabilities_dict)
