from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.validators.atom_array import AtomArrayValidator
from braket.analog_hamiltonian_simulator.rydberg.validators.capabilities_constants import (
    CapabilitiesConstants,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.driving_field import (
    DrivingFieldValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.hamiltonian import HamiltonianValidator
from braket.analog_hamiltonian_simulator.rydberg.validators.physical_field import (
    PhysicalFieldValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.program import ProgramValidator
from braket.analog_hamiltonian_simulator.rydberg.validators.shifting_field import (
    ShiftingFieldValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.waveform import WaveformValidator


def validate_program(program: Program, device_capabilities: CapabilitiesConstants):

    ProgramValidator(capabilities=device_capabilities, **program.dict())
    AtomArrayValidator(capabilities=device_capabilities, **program.setup.atomArray.dict())
    HamiltonianValidator(**program.hamiltonian.dict())
    for d_fields in program.hamiltonian.drivingFields:
        DrivingFieldValidator(capabilities=device_capabilities, **d_fields.dict())
        amplitude = d_fields.amplitude
        phase = d_fields.phase
        detuning = d_fields.detuning
        PhysicalFieldValidator(**amplitude.dict())
        WaveformValidator(capabilities=device_capabilities, **amplitude.sequence.dict())

        PhysicalFieldValidator(**phase.dict())
        WaveformValidator(capabilities=device_capabilities, **phase.sequence.dict())

        PhysicalFieldValidator(**detuning.dict())
        WaveformValidator(capabilities=device_capabilities, **detuning.sequence.dict())
    for s_fields in program.hamiltonian.shiftingFields:
        ShiftingFieldValidator(capabilities=device_capabilities, **s_fields.dict())
        magnitude = s_fields.magnitude
        PhysicalFieldValidator(**magnitude.dict())
        WaveformValidator(capabilities=device_capabilities, **magnitude.sequence.dict())