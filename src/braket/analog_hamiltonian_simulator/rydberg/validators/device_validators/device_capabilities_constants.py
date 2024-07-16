from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants
)
from decimal import Decimal
from pydantic import PositiveInt


class DeviceCapabilitiesConstants(CapabilitiesConstants):
    MAX_SITES: int
    SITE_PRECISION: Decimal
    MAX_FILLED_SITES: int
    
    GLOBAL_TIME_PRECISION: Decimal
    GLOBAL_AMPLITUDE_VALUE_PRECISION: Decimal
    GLOBAL_AMPLITUDE_SLOPE_MAX: Decimal
    GLOBAL_DETUNING_VALUE_PRECISION: Decimal
    GLOBAL_DETUNING_SLOPE_MAX: Decimal
    
    LOCAL_MAGNITUDE_SLOPE_MAX: Decimal
    LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES: Decimal
    LOCAL_TIME_PRECISION: Decimal
    LOCAL_MIN_TIME_SEPARATION: Decimal
    