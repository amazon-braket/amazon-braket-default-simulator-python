from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)

# Default units for simulation
TIME_UNIT = 1e-6  # Time unit for simulation is 1e-6 seconds
SPACE_UNIT = 1e-6  # Space unit for simulation is 1e-6 meters
FIELD_UNIT = 1e6  # Frequency unit for simulation is 1e6 Hz

# All the quantities below are in SI units

# Interaction strength
RYDBERG_INTERACTION_COEF = 5.42e-24

BOUNDING_BOX_SIZE_X = 0.0001
BOUNDING_BOX_SIZE_Y = 0.0001
MIN_BLOCKADE_RADIUS = 1e-06
MAX_TIME = 4e-6

# Constants for Rabi frequency amplitude
MIN_AMPLITUDE = 0
MAX_AMPLITUDE = 25000000.0

# Constants for global detuning
MIN_DETUNING = -125000000.0
MAX_DETUNING = 125000000.0

# Constants for local detuning (shift)
MIN_SHIFT = 0
MAX_SHIFT = 125000000.0
MIN_SHIFT_SCALE = 0.0
MAX_SHIFT_SCALE = 1.0


def capabilities_constants():
    return CapabilitiesConstants(
        BOUNDING_BOX_SIZE_X=BOUNDING_BOX_SIZE_X,
        BOUNDING_BOX_SIZE_Y=BOUNDING_BOX_SIZE_Y,
        MAX_TIME=MAX_TIME,
        GLOBAL_AMPLITUDE_VALUE_MIN=MIN_AMPLITUDE,
        GLOBAL_AMPLITUDE_VALUE_MAX=MAX_AMPLITUDE,
        GLOBAL_DETUNING_VALUE_MIN=MIN_DETUNING,
        GLOBAL_DETUNING_VALUE_MAX=MAX_DETUNING,
        LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN=MIN_SHIFT,
        LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX=MAX_SHIFT,
        MAGNITUDE_PATTERN_VALUE_MIN=MIN_SHIFT_SCALE,
        MAGNITUDE_PATTERN_VALUE_MAX=MAX_SHIFT_SCALE,
    )
