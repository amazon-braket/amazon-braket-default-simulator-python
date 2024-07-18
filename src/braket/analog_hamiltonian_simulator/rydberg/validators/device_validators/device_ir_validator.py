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

from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators.device_atom_arrangement import (
    DeviceAtomArrangementValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators.device_capabilities_constants import (
    DeviceCapabilitiesConstants
)
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators.device_driving_field import (
    DeviceDrivingFieldValidator
)
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators.device_hamiltonian import DeviceHamiltonianValidator
from braket.analog_hamiltonian_simulator.rydberg.validators.device_validators.device_local_detuning import (
    DeviceLocalDetuningValidator
)
from braket.analog_hamiltonian_simulator.rydberg.validators.physical_field import (
    PhysicalFieldValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.program import ProgramValidator
from braket.analog_hamiltonian_simulator.rydberg.validators.times_series import TimeSeriesValidator


def validate_program(program: Program, device_capabilities: DeviceCapabilitiesConstants) -> None:
    """
    Validate the analog Hamiltonian simulation program has only one driving and shifting field,
    and all the sequences have the same last time point.

    Args:
        program (Program): An analog Hamiltonian simulation program
        device_capabilities (CapabilitiesConstants): The capability constants for the simulator
    """

    ProgramValidator(capabilities=device_capabilities, **program.dict())
    DeviceAtomArrangementValidator(capabilities=device_capabilities, **program.setup.ahs_register.dict())
    DeviceHamiltonianValidator(LOCAL_RYDBERG_CAPABILITIES=device_capabilities.LOCAL_RYDBERG_CAPABILITIES, **program.hamiltonian.dict())
    for d_fields in program.hamiltonian.drivingFields:
        DeviceDrivingFieldValidator(capabilities=device_capabilities, **d_fields.dict())
        amplitude = d_fields.amplitude
        phase = d_fields.phase
        detuning = d_fields.detuning
        PhysicalFieldValidator(**amplitude.dict())
        TimeSeriesValidator(capabilities=device_capabilities, **amplitude.time_series.dict())

        PhysicalFieldValidator(**phase.dict())
        TimeSeriesValidator(capabilities=device_capabilities, **phase.time_series.dict())

        PhysicalFieldValidator(**detuning.dict())
        TimeSeriesValidator(capabilities=device_capabilities, **detuning.time_series.dict())
    for s_fields in program.hamiltonian.localDetuning:
        DeviceLocalDetuningValidator(capabilities=device_capabilities, **s_fields.dict())
        magnitude = s_fields.magnitude
        PhysicalFieldValidator(**magnitude.dict())
        TimeSeriesValidator(capabilities=device_capabilities, **magnitude.time_series.dict())
