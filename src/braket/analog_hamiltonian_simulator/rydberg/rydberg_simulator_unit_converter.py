from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import FIELD_UNIT, SPACE_UNIT, TIME_UNIT


def convert_unit(program: Program):
    """
    For a given program, convert the SI units to units that are more
    suitable for local simulator
    """

    setup = program.setup
    sites = setup.atomArray.sites
    new_sites = [[float(site[0]) / SPACE_UNIT, float(site[1]) / SPACE_UNIT] for site in sites]
    new_setup = {"atomArray": {"sites": new_sites, "filling": setup.atomArray.filling}}

    hamiltonian = program.hamiltonian
    driving_fields = hamiltonian.drivingFields
    shifting_fields = hamiltonian.shiftingFields

    new_driving_fields, new_shifting_fields = [], []
    for driving_field in driving_fields:
        new_amplitude = convert_unit_for_field(driving_field.amplitude)
        new_phase = convert_unit_for_field(driving_field.phase, False)
        new_detuning = convert_unit_for_field(driving_field.detuning)
        new_driving_field = {
            "amplitude": new_amplitude,
            "phase": new_phase,
            "detuning": new_detuning,
        }
        new_driving_fields.append(new_driving_field)

    for shifting_field in shifting_fields:
        new_magnitude = convert_unit_for_field(shifting_field.magnitude)
        new_shifting_field = {"magnitude": new_magnitude}
        new_shifting_fields.append(new_shifting_field)

    new_hamiltonian = {"drivingFields": new_driving_fields, "shiftingFields": new_shifting_fields}

    new_program = Program(
        setup=new_setup,
        hamiltonian=new_hamiltonian,
    )
    return new_program


def convert_unit_for_field(field, convertvalues=True):
    """
    For a given field, convert the unit of time from second to microsecond,
    and convert the unit of values from Hz to MHz if `convertvalues`=True
    """
    times = [float(time) / TIME_UNIT for time in field.sequence.times]

    if convertvalues:
        values = [float(value) / FIELD_UNIT for value in field.sequence.values]
    else:
        values = [float(value) for value in field.sequence.values]

    new_field = {"pattern": field.pattern, "sequence": {"times": times, "values": values}}

    return new_field
