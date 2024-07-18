from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants
)
from decimal import Decimal
from typing import Optional


class DeviceCapabilitiesConstants(CapabilitiesConstants):
    MAX_SITES: int
    SITE_PRECISION: Decimal
    MAX_FILLED_SITES: int
    MIN_ROW_DISTANCE: Decimal
    
    GLOBAL_TIME_PRECISION: Decimal
    GLOBAL_AMPLITUDE_VALUE_PRECISION: Decimal
    GLOBAL_AMPLITUDE_SLOPE_MAX: Decimal
    GLOBAL_MIN_TIME_SEPARATION: Decimal
    GLOBAL_DETUNING_VALUE_PRECISION: Decimal
    GLOBAL_DETUNING_SLOPE_MAX: Decimal
    GLOBAL_PHASE_VALUE_MIN: Decimal
    GLOBAL_PHASE_VALUE_MAX: Decimal
    GLOBAL_PHASE_VALUE_PRECISION: Decimal
    
    LOCAL_RYDBERG_CAPABILITIES: bool = False
    LOCAL_MAGNITUDE_SLOPE_MAX: Optional[Decimal]
    LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES: Optional[Decimal]
    LOCAL_TIME_PRECISION: Optional[Decimal]
    LOCAL_MIN_TIME_SEPARATION: Optional[Decimal]
    LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN: Optional[Decimal]
    LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX: Optional[Decimal]
    
    MAGNITUDE_PATTERN_VALUE_MIN: Optional[Decimal]
    MAGNITUDE_PATTERN_VALUE_MAX: Optional[Decimal]
    MAX_NET_DETUNING: Optional[Decimal]
    