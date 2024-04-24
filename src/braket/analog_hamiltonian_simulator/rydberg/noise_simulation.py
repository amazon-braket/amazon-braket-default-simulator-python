import numpy as np

from braket.ahs.atom_arrangement import SiteType
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern

from typing import Dict, List, Tuple
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import AnalogHamiltonianSimulationQuantumTaskResult
from braket.ahs.atom_arrangement import AtomArrangement



from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import (
    Performance, PerformanceLattice, PerformanceRydberg
)

from braket.task_result.task_metadata_v1 import TaskMetadata
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import (
    AnalogHamiltonianSimulationShotMeasurement,
    AnalogHamiltonianSimulationShotMetadata,
    AnalogHamiltonianSimulationShotResult,
    AnalogHamiltonianSimulationTaskResult,
)
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import AnalogHamiltonianSimulationQuantumTaskResult

from braket.devices import LocalSimulator
import multiprocessing as mp
# from braket.aws import AwsQuantumTask


# For noise simulation
from braket.aws import AwsDevice 

qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
capabilities = qpu.properties.paradigm
performance = capabilities.performance


amplitude_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencyRange[-1])
detuning_slew_rate_max = float(capabilities.rydberg.rydbergGlobal.detuningSlewRateMax)
amplitude_slew_rate_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencySlewRateMax)
time_separation_min = float(capabilities.rydberg.rydbergGlobal.timeDeltaMin)
height = float(capabilities.lattice.area.height)
width = float(capabilities.lattice.area.width)
C6 = float(capabilities.rydberg.c6Coefficient)

def apply_site_position_error(
    sites: List[List[float]], 
    site_position_error: float
) -> List[List[float]]:
    """
    Apply site position error to a list of 2D coordinates

    Args:
        sites (List[List[float]]): A list of 2D coordinates
        site_position_error: The site position error, the systematic and pattern-dependent
            error between specified and actual lattice site positions

    Returns:
        erroneous_sites (List[List[float]]): A list of 2D erroneous coordinates
    """
    erroneous_sites = []
    for site in sites:
        erroneous_sites.append(site + site_position_error * np.random.normal(size=2))
        
    return erroneous_sites

def apply_binomial_noise(arr: List[int], p01: float, p10: float):
    """
    Return the noisy array of an otherwise noiseless array subject to a binomial noise

    Args:
        arr (List[int]): An noiseless array
        p01 (float): The probability of mistakenly switching 0 as 1
        p10 (float): The probability of mistakenly switching 1 as 0
        
    Returns:
        noisy_arr (List[int]): The noisy array
    """
    noisy_arr = []
    for val in arr:
        if val == 1:
            # Apply the error of switching 1 as 0
            noisy_arr.append(1 - np.random.binomial(1, p10))
        else:
            # Apply the error of switching 0 as 1
            noisy_arr.append(np.random.binomial(1, p01))
        
    return noisy_arr


def apply_lattice_initialization_errors(
    program: AnalogHamiltonianSimulation,
    performance: Performance, 
    typical_error: bool = True
) -> Tuple[
    List[List[float]], 
    List[int],
    List[int]
]:
    """
    Apply noises for initializing the atomic lattice

    Args:
        program (AnalogHamiltonianSimulation): An AHS program
        performance (Performance): The parameters determining the limitations of the Rydberg device
        typical_error (bool): If true, apply the typical values for the parameters, otherwise, apply 
            the worst-case values for the parameters. Default True.
        
    Returns:
        erroneous_sites (List[List[float]]): A list of 2D erroneous coordinates
        erroneous_filling (List[int]): A list of erroneous filling
        pre_seq (List[int]): The pre-sequence
    """
    
    # program_ir = program.to_ir()
    
    site_position_err = float(performance.lattice.sitePositionError)
    ground_prep_err = float(performance.rydberg.rydbergGlobal.groundPrepError)
    if typical_error:
        filling_err = float(performance.lattice.vacancyErrorTypical)
        vacancy_err = float(performance.lattice.vacancyErrorTypical)
        atom_det_false_positive = float(performance.lattice.atomDetectionErrorFalsePositiveTypical)
        atom_det_false_negative = float(performance.lattice.atomDetectionErrorFalseNegativeTypical)
    else:
        filling_err = float(performance.lattice.vacancyErrorWorst)
        vacancy_err = float(performance.lattice.vacancyErrorWorst)
        atom_det_false_positive = float(performance.lattice.atomDetectionErrorFalsePositiveWorst)
        atom_det_false_negative = float(performance.lattice.atomDetectionErrorFalseNegativeWorst)        
    
    # sites = [[float(site[0]), float(site[1])] for site in program_ir.setup.ahs_register.sites]
    # filling = program_ir.setup.ahs_register.filling
    sites = [[x, y] for (x, y) in zip(program.register.coordinate_list(0), program.register.coordinate_list(1))]
    filling = program.to_ir().setup.ahs_register.filling
    
    erroneous_sites = apply_site_position_error(sites, site_position_err)
    erroneous_filling = apply_binomial_noise(filling, filling_err, vacancy_err)
    pre_seq = apply_binomial_noise(erroneous_filling, atom_det_false_negative, atom_det_false_positive)

    erroneous_filling = apply_binomial_noise(erroneous_filling, 0, ground_prep_err)
    
    return erroneous_sites, erroneous_filling, pre_seq    


def apply_detuning_errors(
    detuning: TimeSeries,
    filling: List[int],
    steps: int,
    detuning_error: float,
    detuning_inhomogeneity: float
) -> Tuple[TimeSeries, ShiftingField]:
    """
    Apply noises to the detuning

    Args:
        detuning (TimeSeries): The time series for the detuning
        filling (List[int]): The filling of the atom array
        steps (int): The number of time steps in the simulation
        detuning_error (float): The detuning error
        detuning_inhomogeneity (float): The detuning inhomogeneity
        
    Returns:
        noisy_detuning (TimeSeries): The time series of the noisy detuning
        shift (ShiftingField): The shifting field used to simulate the detuning inhomogeneity
    """    
    
    detuning_times = detuning.time_series.times()
    detuning_values = detuning.time_series.values()
    
    noisy_detuning_times = np.linspace(0, detuning_times[-1], steps)
    noisy_detuning_values = np.interp(noisy_detuning_times, detuning_times, detuning_values)

    # Apply the detuning error
    noisy_detuning_values += detuning_error * np.random.normal(size=len(noisy_detuning_values))
    
    noisy_detuning = TimeSeries.from_lists(noisy_detuning_times, noisy_detuning_values)
    
    # Apply detuningInhomogeneity        
    h = Pattern([1 for _ in filling])
    detuning_local = TimeSeries.from_lists(noisy_detuning_times, 
                                           detuning_inhomogeneity * np.random.normal(
                                               size=len(noisy_detuning_times)
                                           )
                                          )
    
    # h = Pattern([np.random.uniform() for _ in filling])
    # detuning_local = TimeSeries()
    # detuning_local.put(0.0, detuning_inhomogeneity)
    # detuning_local.put(detuning_times[-1], detuning_inhomogeneity)
    
    # Assemble the local shift
    shift = ShiftingField(
        magnitude=Field(
            time_series=detuning_local,
            pattern=h
        )
    )
    
    return noisy_detuning, shift


def determine_a_b_c(t1, t2, v1, v2, f):
    a = 3 * (v1 + f * v1 + v2 - f * v2)/(t1 - t2)**2
    c = (t2 * v1 * ((2 + 3 * f) * t1 + t2) + t1 * v2 * (t1 + (2 - 3 * f) * t2))/(t1 - t2)**2
    b = (v2 - c - a * t2**2) / t2
        
    return a, b, c
    
    
    
def apply_amplitude_errors(
    amplitude: TimeSeries,
    steps: int,
    rabi_error_rel: float,
    rabi_ramp_correction: List,
    amplitude_max = amplitude_max
) -> TimeSeries:
    """
    Apply noises to the amplitude

    Args:
        amplitude (TimeSeries): The time series for the amplitude
        steps (int): The number of time steps in the simulation
        rabi_error_rel (float): The amplitude error as a relative value
        rabi_ramp_correction (List): The dynamic correction to ramped amplitude 
            as relative values
        
    Returns:
        noisy_amplitude (TimeSeries): The time series of the noisy amplitude
    """    
    
    amplitude_times = amplitude.time_series.times()
    amplitude_values = amplitude.time_series.values()
    
    # Rewrite the rabi_ramp_correction as a function of slopes
    rabi_ramp_correction_slopes = [amplitude_max / float(corr.rampTime)
        for corr in rabi_ramp_correction
    ]
    rabi_ramp_correction_fracs = [float(corr.rabiCorrection)
        for corr in rabi_ramp_correction
    ]    
    rabi_ramp_correction_slopes = rabi_ramp_correction_slopes[::-1]
    rabi_ramp_correction_fracs = rabi_ramp_correction_fracs[::-1]
    # print(rabi_ramp_correction_slopes)
    # print(rabi_ramp_correction_fracs)
        
    noisy_amplitude_times = np.linspace(0, amplitude_times[-1], steps)
    noisy_amplitude_values = []
    
    # First apply the rabi ramp correction
    for ind in range(len(amplitude_times)):
        if ind == 0:
            continue
            
        # First determine the correction factor from the slope
        t1, t2 = amplitude_times[ind-1], amplitude_times[ind]
        v1, v2 = amplitude_values[ind-1], amplitude_values[ind]
        slope = (v2 - v1) / (t2 - t1)
        # print(f"ind, slope, sign(slope) = {ind}, {slope}, {np.sign(slope)}")
        slope_ind = np.searchsorted(
            rabi_ramp_correction_slopes,
            np.abs(slope)
        )
        
        if slope_ind == 0:
            frac = 1.0
        elif slope_ind == len(rabi_ramp_correction_slopes):
            # raise ValueError("The amplitude slew rate is larger than the maximum allowed value")
            frac = 1.0
        else:
            fracs_diff = rabi_ramp_correction_fracs[slope_ind] - rabi_ramp_correction_fracs[slope_ind-1]
            slope_diff = rabi_ramp_correction_slopes[slope_ind] - rabi_ramp_correction_slopes[slope_ind-1]
            frac = rabi_ramp_correction_fracs[slope_ind-1] + (np.abs(slope) - rabi_ramp_correction_slopes[slope_ind-1]) * fracs_diff / slope_diff
            frac *= np.sign(slope)
        
        # print(f"slope_ind, frac = {slope_ind}, {frac}")
        
        # Next, determine the coefficients for the quadratic correction
        if frac == 1.0:
            a, b, c = 0, 0, v2
        else:
            a, b, c = determine_a_b_c(t1, t2, v1, v2, frac)
        
        # Finally, put values into noisy_amplitude_values
        for t in noisy_amplitude_times:
            if t1 <= t and t <= t2:
                noisy_amplitude_values.append(a * t**2 + b * t + c)

                
    # Next apply amplitude error
    rabi_errors = 1 + rabi_error_rel * np.random.normal(size=len(noisy_amplitude_values))
    noisy_amplitude_values = np.multiply(noisy_amplitude_values, rabi_errors)
    noisy_amplitude_values = [max(0, value) for value in noisy_amplitude_values] # amplitude has to be non-negative
                
    noisy_amplitude = TimeSeries.from_lists(noisy_amplitude_times, noisy_amplitude_values)
    
    return noisy_amplitude


def apply_rydberg_noise(
    program: AnalogHamiltonianSimulation,
    performance: Performance,
    steps: int,
):
    # program_ir = program.to_ir()
    
    detuning_error = float(performance.rydberg.rydbergGlobal.detuningError)
    detuning_inhomogeneity = float(performance.rydberg.rydbergGlobal.detuningInhomogeneity)
    
    noisy_detuning, shift = apply_detuning_errors(program.hamiltonian.detuning,
                                                  program.to_ir().setup.ahs_register.filling,
                                                  steps, 
                                                  detuning_error, 
                                                  detuning_inhomogeneity
                                                 )    
    
    rabi_error_rel = float(performance.rydberg.rydbergGlobal.rabiFrequencyGlobalErrorRel)
    rabi_ramp_correction = performance.rydberg.rydbergGlobal.rabiAmplitudeRampCorrection
    noisy_amplitude = program.hamiltonian.amplitude
    # noisy_amplitude = apply_amplitude_errors(program.hamiltonian.amplitude,
    #                                          steps,
    #                                          rabi_error_rel,
    #                                          rabi_ramp_correction
    #                                         )
           
    noisy_drive = DrivingField(amplitude = noisy_amplitude, 
                         detuning = noisy_detuning,
                         phase = program.hamiltonian.phase
                        )
    
    return noisy_drive, shift

def get_shot_measurement(
    args,
    simulator = LocalSimulator("braket_ahs")
    # program: AnalogHamiltonianSimulation,
    # noise_model: Performance,
    # steps: int = 100,    
):
    program, noise_model, steps = args[0], args[1], args[2]
    # sites, fillings, preseq = apply_lattice_noise(program, noise_model.lattice)
    # drive, shift = apply_rydberg_noise(program, noise_model.rydberg)
    sites, fillings, preseq = apply_lattice_initialization_errors(program, noise_model)
    drive, shift = apply_rydberg_noise(program, noise_model, steps)

    # Assemble the noisy program
    register = AtomArrangement()
    for (site, filling) in zip(sites, fillings):
        if filling == 1:
            register.add(site)
        else:
            register.add(site, site_type=SiteType.VACANT)

    noisy_program = AnalogHamiltonianSimulation(
        register=register, 
        hamiltonian=drive+shift
    )

    result = simulator.run(noisy_program, shots=1, steps=steps).result()


    # preseq = result.measurements[0].pre_sequence
    postseq = result.measurements[0].post_sequence

    # Aplly groundDetectionError and rydbergDetectionError
    grd_det_error = float(performance.rydberg.rydbergGlobal.groundDetectionError)
    ryd_det_error = float(performance.rydberg.rydbergGlobal.rydbergDetectionError)

    new_postseq = apply_binomial_noise(postseq, ryd_det_error, grd_det_error)


    shot_measurement = AnalogHamiltonianSimulationShotMeasurement(
        shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
        shotResult=AnalogHamiltonianSimulationShotResult(
            preSequence=preseq, postSequence=new_postseq
        ),
    )       
    return shot_measurement

def ahs_noise_simulation(
    program: AnalogHamiltonianSimulation,
    noise_model: Performance,
    shots: int = 1000,
    steps: int = 100,
):
    
    task_metadata = TaskMetadata(
        id="rydberg",
        shots=shots,
        deviceId="rydbergLocalSimulator",
    )            
    
    # measurements = [get_shot_measurement([program, noise_model, steps]) for _ in range(shots)]

    with mp.Pool(processes=mp.cpu_count(), initializer=np.random.seed) as p:
        measurements = p.map(get_shot_measurement, [[program, noise_model, steps] for _ in range(shots)])
    
    ahs_task_result = AnalogHamiltonianSimulationTaskResult(
        taskMetadata=task_metadata, measurements=measurements
    )
    
    return AnalogHamiltonianSimulationQuantumTaskResult(ahs_task_result, additional_metadata=None)
