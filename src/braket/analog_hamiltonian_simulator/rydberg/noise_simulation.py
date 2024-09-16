import numpy as np

from braket.ahs.atom_arrangement import SiteType
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.local_detuning import LocalDetuning
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern

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
# performance = capabilities.performance


amplitude_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencyRange[-1])
# detuning_slew_rate_max = float(capabilities.rydberg.rydbergGlobal.detuningSlewRateMax)
# amplitude_slew_rate_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencySlewRateMax)
# time_separation_min = float(capabilities.rydberg.rydbergGlobal.timeDeltaMin)
# height = float(capabilities.lattice.area.height)
# width = float(capabilities.lattice.area.width)
# C6 = float(capabilities.rydberg.c6Coefficient)

def apply_site_position_error(
    sites: list[list[float]], 
    site_position_error: float
) -> list[list[float]]:
    """
    Apply site position error to a list of 2D coordinates

    Args:
        sites (list[list[float]]): A list of 2D coordinates
        site_position_error: The site position error, the systematic and pattern-dependent
            error between specified and actual lattice site positions

    Returns:
        erroneous_sites (list[list[float]]): A list of 2D erroneous coordinates
    """
    erroneous_sites = []
    for site in sites:
        erroneous_sites.append(site + site_position_error * np.random.normal(size=2))
        
    return erroneous_sites

def apply_binomial_noise(arr: list[int], p01: float, p10: float):
    """
    Return the noisy array of an otherwise noiseless array subject to a binomial noise

    Args:
        arr (list[int]): An noiseless array
        p01 (float): The probability of mistakenly switching 0 as 1
        p10 (float): The probability of mistakenly switching 1 as 0
        
    Returns:
        noisy_arr (list[int]): The noisy array
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


from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import Performance

def apply_lattice_initialization_errors(
    program: AnalogHamiltonianSimulation,
    performance: Performance, 
    typical_error: bool = True
) -> tuple[
    list[list[float]], 
    list[int],
    list[int]
]:
    """
    Apply noises for initializing the atomic lattice

    Args:
        program (AnalogHamiltonianSimulation): An AHS program
        performance (Performance): The parameters determining the limitations of the Rydberg device
        typical_error (bool): If true, apply the typical values for the parameters, otherwise, apply 
            the worst-case values for the parameters. Default True.
        
    Returns:
        erroneous_sites (list[list[float]]): A list of 2D erroneous coordinates
        erroneous_filling (list[int]): A list of erroneous filling
        pre_seq (list[int]): The pre-sequence
    """
        
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
    
    sites = [[x, y] for (x, y) in zip(program.register.coordinate_list(0), program.register.coordinate_list(1))]
    filling = program.to_ir().setup.ahs_register.filling
    
    erroneous_sites = apply_site_position_error(sites, site_position_err)
    erroneous_filling = apply_binomial_noise(filling, filling_err, vacancy_err)
    pre_seq = apply_binomial_noise(erroneous_filling, atom_det_false_negative, atom_det_false_positive)

    erroneous_filling = apply_binomial_noise(erroneous_filling, 0, ground_prep_err)
    
    return erroneous_sites, erroneous_filling, pre_seq    

import scipy

import scipy

def apply_amplitude_errors(
    amplitude: TimeSeries,
    steps: int,
    rabi_error_rel: float,
    rabi_ramp_correction: list,
    amplitude_max = amplitude_max
) -> TimeSeries:
    """
    Apply noises to the amplitude

    Args:
        amplitude (TimeSeries): The time series for the amplitude
        steps (int): The number of time steps in the simulation
        rabi_error_rel (float): The amplitude error as a relative value
        rabi_ramp_correction (list): The dynamic correction to ramped amplitude 
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
    
    # Helper function to find the correction factor for a given slope
    get_frac = scipy.interpolate.interp1d(rabi_ramp_correction_slopes, 
                                   rabi_ramp_correction_fracs, 
                                   bounds_error=False, 
                                   fill_value="extrapolate"
                                  )
        
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
        if np.abs(slope) > 0:
            frac = get_frac(np.abs(slope)) * np.sign(slope)        
        else:
            frac = 1.0
        
        # Next, determine the coefficients for the quadratic correction
        if frac >= 1.0:
            a, b, c = 0, 0, v2
        else:
            # Determine the coefficients for the quadratic correction
            # of the form f(t) = a*t^2 + b * t + c 
            # such that f(t1) = v1 and f(t2) = v2 and 
            # a/3*(t2^3-t1^3) + b/2*(t2^2-t1^2) + c(t2-t1) = frac * (t2-t1) * (v2-v1)/2
            
            a = 3 * (v1 + frac * v1 + v2 - frac * v2)/(t1 - t2)**2
            c = (t2 * v1 * ((2 + 3 * frac) * t1 + t2) + t1 * v2 * (t1 + (2 - 3 * frac) * t2))/(t1 - t2)**2
            b = (v2 - c - a * t2**2) / t2    
        
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

def apply_detuning_errors(
    detuning: TimeSeries,
    filling: list[int],
    steps: int,
    detuning_error: float,
    detuning_inhomogeneity: float
) -> tuple[TimeSeries, LocalDetuning]:
    """
    Apply noises to the detuning

    Args:
        detuning (TimeSeries): The time series for the detuning
        filling (list[int]): The filling of the atom array
        steps (int): The number of time steps in the simulation
        detuning_error (float): The detuning error
        detuning_inhomogeneity (float): The detuning inhomogeneity
        
    Returns:
        noisy_detuning (TimeSeries): The time series of the noisy detuning
        local_detuning (LocalDetuning): The local detuning used to simulate the detuning inhomogeneity
    """    
    
    detuning_times = detuning.time_series.times()
    detuning_values = detuning.time_series.values()
    
    noisy_detuning_times = np.linspace(0, detuning_times[-1], steps)
    noisy_detuning_values = np.interp(noisy_detuning_times, detuning_times, detuning_values)

    # Apply the detuning error
    noisy_detuning_values += detuning_error * np.random.normal(size=len(noisy_detuning_values))
    
    noisy_detuning = TimeSeries.from_lists(noisy_detuning_times, noisy_detuning_values)
    
    # Apply detuning inhomogeneity        
    h = Pattern([np.random.rand() for _ in filling])
    detuning_local = TimeSeries.from_lists(noisy_detuning_times, 
                                           detuning_inhomogeneity * np.ones(
                                               len(noisy_detuning_times)
                                           )
                                          )

    # Assemble the local detuning
    local_detuning = LocalDetuning(
        magnitude=Field(
            time_series=detuning_local,
            pattern=h
        )
    )
    
    return noisy_detuning, local_detuning


def apply_rydberg_noise(
    program: AnalogHamiltonianSimulation,
    performance: Performance,
    steps: int,
):
    
    detuning_error = float(performance.rydberg.rydbergGlobal.detuningError)
    detuning_inhomogeneity = float(performance.rydberg.rydbergGlobal.detuningInhomogeneity)
    
    noisy_detuning, local_detuning = apply_detuning_errors(program.hamiltonian.detuning,
                                                  program.to_ir().setup.ahs_register.filling,
                                                  steps, 
                                                  detuning_error, 
                                                  detuning_inhomogeneity
                                                 )    
    
    rabi_error_rel = float(performance.rydberg.rydbergGlobal.rabiFrequencyGlobalErrorRel)
    rabi_ramp_correction = performance.rydberg.rydbergGlobal.rabiAmplitudeRampCorrection
    noisy_amplitude = apply_amplitude_errors(program.hamiltonian.amplitude,
                                             steps,
                                             rabi_error_rel,
                                             rabi_ramp_correction
                                            )
           
    noisy_drive = DrivingField(amplitude = noisy_amplitude, 
                         detuning = noisy_detuning,
                         phase = program.hamiltonian.phase
                        )
    
    return noisy_drive, local_detuning

def apply_measurement_errors(postseq: list[int], performance: Performance) -> list[int]:
    grd_det_error = float(performance.rydberg.rydbergGlobal.groundDetectionError)
    ryd_det_error = float(performance.rydberg.rydbergGlobal.rydbergDetectionError)

    return apply_binomial_noise(postseq, ryd_det_error, grd_det_error)    

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

    postseq = result.measurements[0].post_sequence
    new_postseq = apply_measurement_errors(postseq, noise_model)

    # # Aplly groundDetectionError and rydbergDetectionError
    # grd_det_error = float(performance.rydberg.rydbergGlobal.groundDetectionError)
    # ryd_det_error = float(performance.rydberg.rydbergGlobal.rydbergDetectionError)

    # new_postseq = apply_binomial_noise(postseq, ryd_det_error, grd_det_error)


    shot_measurement = AnalogHamiltonianSimulationShotMeasurement(
        shotMetadata=AnalogHamiltonianSimulationShotMetadata(shotStatus="Success"),
        shotResult=AnalogHamiltonianSimulationShotResult(
            preSequence=preseq, postSequence=new_postseq
        ),
    )       
    return shot_measurement

def get_shot_measurement_tn(
    args,
    simulator = LocalSimulator("braket_ahs_tn"),
    # program: AnalogHamiltonianSimulation,
    # noise_model: Performance,
    # steps: int = 100,    
):
    program, noise_model, steps, blockade_radius, max_bond_dim, solver = args[0], args[1], args[2], args[3], args[4], args[5]

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

    result = simulator.run(noisy_program, shots=1, steps=steps, blockade_radius=blockade_radius, max_bond_dim=max_bond_dim, solver=solver).result()

    postseq = result.measurements[0].post_sequence
    new_postseq = apply_measurement_errors(postseq, noise_model)

    # # Aplly groundDetectionError and rydbergDetectionError
    # grd_det_error = float(performance.rydberg.rydbergGlobal.groundDetectionError)
    # ryd_det_error = float(performance.rydberg.rydbergGlobal.rydbergDetectionError)

    # new_postseq = apply_binomial_noise(postseq, ryd_det_error, grd_det_error)


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
        deviceId="ahs_noise_simulation",
    )            
    
    with mp.Pool(processes=mp.cpu_count(), initializer=np.random.seed) as p:
        measurements = p.map(get_shot_measurement, [[program, noise_model, steps] for _ in range(shots)])
    
    ahs_task_result = AnalogHamiltonianSimulationTaskResult(
        taskMetadata=task_metadata, measurements=measurements
    )
    
    return AnalogHamiltonianSimulationQuantumTaskResult.from_object(ahs_task_result)

def convert_ir_program_back(program_ir):
    program_ir_dict = program_ir.dict()

    ahs_register = program_ir_dict['setup']['ahs_register']
    hamiltonian = program_ir_dict['hamiltonian']

    # for register
    sites = ahs_register['sites']
    fillings = ahs_register['filling']
    register = AtomArrangement()

    for site, filling in zip(sites, fillings):
        if filling == 1:
            register.add((float(site[0]), float(site[1])))
        else:
            register.add((float(site[0]), float(site[1])), site_type=SiteType.VACANT)

    # for drive
    amplitude_values = hamiltonian['drivingFields'][0]['amplitude']["time_series"]["values"]
    amplitude_times = hamiltonian['drivingFields'][0]['amplitude']["time_series"]["times"]
    detuning_values = hamiltonian['drivingFields'][0]['detuning']["time_series"]["values"]
    detuning_times = hamiltonian['drivingFields'][0]['detuning']["time_series"]["times"]
    phase_values = hamiltonian['drivingFields'][0]['phase']["time_series"]["values"]
    phase_times = hamiltonian['drivingFields'][0]['phase']["time_series"]["times"]

    amplitude_values = [float(i) for i in amplitude_values]
    amplitude_times = [float(i) for i in amplitude_times]
    detuning_values = [float(i) for i in detuning_values]
    detuning_times = [float(i) for i in detuning_times]
    phase_values = [float(i) for i in phase_values]
    phase_times = [float(i) for i in phase_times]


    amplitude = TimeSeries.from_lists(amplitude_times, amplitude_values)
    detuning = TimeSeries.from_lists(detuning_times, detuning_values)
    phase = TimeSeries.from_lists(phase_times, phase_values)

    drive = DrivingField(
        amplitude=amplitude,
        phase=phase,
        detuning=detuning
    )

    if len(hamiltonian['localDetuning']) > 0:
    
        local_detuning_values = hamiltonian['localDetuning'][0]['magnitude']["time_series"]["values"]
        local_detuning_times = hamiltonian['localDetuning'][0]['magnitude']["time_series"]["times"]
        pattern = hamiltonian['localDetuning'][0]['magnitude']['pattern']

        local_detuning_values = [float(i) for i in local_detuning_values]
        local_detuning_times = [float(i) for i in local_detuning_times]
        pattern = [float(i) for i in pattern]

        local_detuning = LocalDetuning.from_lists(local_detuning_times, local_detuning_values, pattern)
        
        hamiltonian=drive + local_detuning
    else:
        hamiltonian=drive

    program = AnalogHamiltonianSimulation(
        hamiltonian=hamiltonian,
        register=register
    )   
    
    return program
