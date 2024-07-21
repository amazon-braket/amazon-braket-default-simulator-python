from pydantic.v1.class_validators import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.hamiltonian import HamiltonianValidator


class DeviceHamiltonianValidator(HamiltonianValidator):
    LOCAL_RYDBERG_CAPABILITIES: bool = False

    @root_validator(pre=True, skip_on_failure=True)
    def max_zero_local_detuning(cls, values):
        LOCAL_RYDBERG_CAPABILITIES = values["LOCAL_RYDBERG_CAPABILITIES"]
        local_detuning = values.get("localDetuning", [])
        if not LOCAL_RYDBERG_CAPABILITIES:
            if len(local_detuning) > 0:
                raise ValueError(
                    f"Local detuning cannot be specified; \
{len(local_detuning)} are given. Specifying local \
detuning is an experimental capability, use Braket Direct to request access."
                )
        return values
