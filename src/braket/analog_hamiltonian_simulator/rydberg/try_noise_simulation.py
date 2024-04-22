from pprint import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing as mp
from typing import List, Tuple

from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries


from braket.aws import AwsDevice 

from braket.analog_hamiltonian_simulator.rydberg.noise_simulation import ahs_noise_simulation

qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
capabilities = qpu.properties.paradigm

# pp(capabilities.rydberg.rydbergGlobal.dict())

amplitude_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencyRange[-1])
detuning_slew_rate_max = float(capabilities.rydberg.rydbergGlobal.detuningSlewRateMax)
amplitude_slew_rate_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencySlewRateMax)
time_separation_min = float(capabilities.rydberg.rydbergGlobal.timeDeltaMin)
height = float(capabilities.lattice.area.height)
width = float(capabilities.lattice.area.width)
C6 = float(capabilities.rydberg.c6Coefficient)

atom_separation = 6.7e-6 # The separation of the two atoms used throughout the notebook

def create_evolve_bell_states(
    atom_separation = atom_separation, 
    amplitude_max = amplitude_max,
    t_ramp_amplitude = 5e-8,
    amplitude_area = 0.0,
    detuning_slew_rate = detuning_slew_rate_max,
    time_separation_min = time_separation_min,
    # amplitude_slew_rate = amplitude_slew_rate_max * 0.8,
    amplitude_slew_rate = amplitude_slew_rate_max,
    if_show_global_drive = False,
    if_parallel = False,
    height = height,
    width = width,
    patch_separation = 24e-6,
) -> AnalogHamiltonianSimulation:
    """
    Return an AHS program to create and evolve a Bell state with Rydberg atoms

    Args:
        atom_separation (float): The separation of the two atoms
        amplitude_max (float): The maximum amplitude in the program
        t_ramp_amplitude (float): The time of ramping up amplitude for creating Bell state
        amplitude_area (float): The area of the amplitude for the evolution
        detuning_slew_rate (float): The slew rate for detuning in the program
        amplitude_slew_rate (float): The slew rate for amplitude in the program
        if_show_global_drive (bool): If true, a figure for global drive will be shown


    Returns:
        AnalogHamiltonianSimulation: The AHS program for creating and evolving a Bell state
    """    
    
    detuning_max = C6/(atom_separation**6)
    t_ramp_detuning = detuning_max / detuning_slew_rate
    t_ramp_detuning = max(t_ramp_detuning, time_separation_min)
    
    # Define register
    coords = [(0, 0), (0, atom_separation)]
    if if_parallel:
        # Extend along y direction
        n_height = int(float(height) // (atom_separation + patch_separation))
        if n_height * (atom_separation + patch_separation) + atom_separation < height:
            n_height += 1
            
        for i in range(n_height-1):
            coords.append((0, patch_separation * (i+1)))
            coords.append((0, atom_separation + patch_separation * (i+1)))
            
        # Extend along x direction
        n_width = int(float(width) // patch_separation)
        if n_width * patch_separation < width:
            n_width += 1
        
        for i in range(n_width-1):
            for j in range(n_height):
                coords.append((patch_separation * (i+1), patch_separation * j))
                coords.append((patch_separation * (i+1), atom_separation + patch_separation * j))
    
    register = AtomArrangement()
    for coord in coords:
        register.add(coord)

    # Prepare the Bell state
    amplitude = TimeSeries.trapezoidal_signal(np.pi/np.sqrt(2), amplitude_max, amplitude_slew_rate, time_separation_min=time_separation_min)
    detuning = TimeSeries.constant_like(amplitude, 0.0)
    phase = TimeSeries.constant_like(amplitude, 0.0)
    
    # Ramp up the detuning
    t_prep = amplitude.times()[-1]
    amplitude.put(t_prep + t_ramp_detuning, 0.0)
    detuning.put(t_prep + t_ramp_detuning, detuning_max)
    phase.put(t_prep + t_ramp_detuning, 0.0)
    
    # Evolve
    if amplitude_area > 0:
        amplitude_evolve = TimeSeries.trapezoidal_signal(amplitude_area, amplitude_max, amplitude_slew_rate, time_separation_min=time_separation_min)
        amplitude = amplitude.stitch(amplitude_evolve)
        detuning = detuning.stitch(TimeSeries.constant_like(amplitude_evolve, detuning_max))
        phase = phase.stitch(TimeSeries.constant_like(amplitude_evolve, 0.0))
    
    
    drive = DrivingField(amplitude=amplitude, phase=phase, detuning=detuning)
    
    # if if_show_global_drive:
    #     show_global_drive(drive)
                
    program = AnalogHamiltonianSimulation(
        register = register,
        hamiltonian = drive
    )
    
    return program



amplitude_areas = [np.pi/24 * i for i in range(19)]
programs = [create_evolve_bell_states(amplitude_area = amplitude_area) for amplitude_area in amplitude_areas]

performance = capabilities.performance

# noisy_result = ahs_noise_simulation(programs[0], performance, shots=100, steps = 100)

if __name__ == "__main__":
    noisy_result = ahs_noise_simulation(programs[0], performance, shots=100, steps = 100)

#     print(noisy_result)
# noisy_results = [ahs_noise_simulation(program, performance, shots=100, steps = 100) for program in programs]