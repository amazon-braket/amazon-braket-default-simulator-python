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

"""
GPU single operation strategy for quantum simulations.

This module provides GPU acceleration for quantum circuit execution using
efficient ping-pong buffering, cuda.jit kernels, and advanced memory management
for scaling to high qubit counts.
"""

import numpy as np
from numba import cuda
import math
import warnings

from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation_strategies import single_operation_strategy
from braket.default_simulator.linalg_utils import (
    DIAGONAL_GATES,
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
    _OPTIMAL_THREADS_PER_BLOCK,
    _apply_cnot_gpu_inplace,
    _apply_controlled_phase_shift_gpu_inplace,
    _apply_diagonal_gate_gpu_inplace,
    _apply_single_qubit_gate_gpu_inplace,
    _apply_swap_gpu_inplace,
    _apply_two_qubit_gate_gpu_inplace,
    _should_use_gpu,
)
from braket.default_simulator.persistent_gpu_state_manager import (
    get_persistent_gpu_manager,
)
from braket.default_simulator.gpu_memory_optimizer import GPUMemoryOptimizer


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply operations to state vector using GPU acceleration with advanced memory management."""
    memory_info = _check_gpu_memory_availability(state.size)
    use_gpu = (_GPU_AVAILABLE and 
               qubit_count >= 8 and 
               state.size >= 256 and 
               memory_info['can_fit'])
    
    if not use_gpu:
        if not _GPU_AVAILABLE:
            print(f"Using CPU for {qubit_count} qubits ({state.size} elements): GPU not available")
        elif not memory_info['can_fit']:
            print(f"Using CPU for {qubit_count} qubits ({state.size} elements): Insufficient GPU memory")
            print(f"Required: {memory_info['required_gb']:.2f} GB, Available: {memory_info['available_gb']:.2f} GB")
        else:
            print(f"Using CPU for {qubit_count} qubits ({state.size} elements): Below GPU threshold")
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    print(f"Using GPU for {qubit_count} qubits ({state.size} elements)")
    print(f"GPU memory usage: {memory_info['required_gb']:.2f}/{memory_info['available_gb']:.2f} GB")
    
    result = _execute_gpu_operations(state, qubit_count, operations, memory_info)
    
    clear_gpu_caches()
    
    return result


def clear_gpu_caches():
    """Clear all GPU caches and reset state."""
    if not _GPU_AVAILABLE:
        return
        
    gpu_manager = get_persistent_gpu_manager()
    if gpu_manager is not None:
        gpu_manager.force_cleanup()
    
    import gc
    gc.collect()
    
    cuda.synchronize()


def _check_gpu_memory_availability(state_size: int) -> dict:
    """Check if GPU has sufficient memory for the given state size.
    
    Args:
        state_size: Number of complex128 elements in the state vector
        
    Returns:
        Dictionary with memory availability information
    """
    if not _GPU_AVAILABLE:
        return {'can_fit': False, 'required_gb': 0, 'available_gb': 0}
    
    import gc
    gc.collect()
    cuda.synchronize()
    
    free_bytes, total_bytes = cuda.current_context().get_memory_info()
    
    bytes_per_complex = 16
    state_bytes = state_size * bytes_per_complex
    required_bytes = state_bytes * 2.5
    
    safety_margin = total_bytes * 0.15
    effective_available = free_bytes - safety_margin
    can_fit = required_bytes <= effective_available
    
    return {
        'can_fit': can_fit,
        'required_gb': required_bytes / (1024**3),
        'available_gb': effective_available / (1024**3),
        'total_gb': total_bytes / (1024**3),
        'free_gb': free_bytes / (1024**3)
    }


def _get_optimal_launch_config(total_size: int, qubit_count: int) -> tuple[int, int]:
    """Calculate optimal GPU launch configuration using grid-stride loop strategy.
    
    Args:
        total_size: Total number of elements to process
        qubit_count: Number of qubits in the system
        
    Returns:
        Tuple of (blocks_per_grid, threads_per_block)
    """
    if total_size >= 2**22:
        threads_per_block = 256
    elif total_size >= 2**18:
        threads_per_block = 512
    else:
        threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    
    if total_size >= 2**20:
        blocks_per_grid = min(32 * 80, _MAX_BLOCKS_PER_GRID)
        min_blocks_needed = (total_size + threads_per_block * 32 - 1) // (threads_per_block * 32)
        blocks_per_grid = max(blocks_per_grid, min(min_blocks_needed, _MAX_BLOCKS_PER_GRID))
    else:
        blocks_needed = (total_size + threads_per_block - 1) // threads_per_block
        blocks_per_grid = min(blocks_needed, _MAX_BLOCKS_PER_GRID)
    
    return blocks_per_grid, threads_per_block


def _execute_gpu_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation], memory_info: dict
) -> np.ndarray:
    """Execute operations on GPU with advanced memory management and optimized configurations."""
    gpu_manager = get_persistent_gpu_manager()
    
    if gpu_manager is not None and qubit_count >= 22:
        result_buffer, temp_buffer = gpu_manager.get_persistent_state(state, force_refresh=False)
        use_persistent = True
        stream = None
    else:
        stream = cuda.stream()
        
        if qubit_count >= 20:
            if state.flags.c_contiguous:
                pinned_state = cuda.pinned_array_like(state)
                pinned_state[:] = state
                result_buffer = cuda.to_device(pinned_state, stream=stream)
            else:
                contiguous_state = np.ascontiguousarray(state)
                pinned_state = cuda.pinned_array_like(contiguous_state)
                pinned_state[:] = contiguous_state
                result_buffer = cuda.to_device(pinned_state, stream=stream)
        else:
            if state.flags.c_contiguous:
                result_buffer = cuda.to_device(state, stream=stream)
            else:
                result_buffer = cuda.to_device(np.ascontiguousarray(state), stream=stream)
        
        temp_buffer = cuda.device_array_like(result_buffer)
        stream.synchronize()
        use_persistent = False
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(getattr(op, "_ctrl_modifiers", []))
        
        needs_swap = _apply_operation_gpu(
            op, result_buffer, temp_buffer, qubit_count,
            targets[num_ctrl:], targets[:num_ctrl], getattr(op, "_ctrl_modifiers", [])
        )
        
        if needs_swap:
            result_buffer, temp_buffer = temp_buffer, result_buffer
    
    if stream:
        stream.synchronize()
    else:
        cuda.synchronize()
    
    if use_persistent:
        return gpu_manager.get_result_array(result_buffer, use_zero_copy=True)
    else:
        if qubit_count >= 20:
            pinned_result = cuda.pinned_array_like(state)
            result_buffer.copy_to_host(ary=pinned_result, stream=stream)
            if stream:
                stream.synchronize()
            return pinned_result.copy()
        else:
            return result_buffer.copy_to_host()


def _apply_operation_gpu(
    op: GateOperation,
    result_buffer: cuda.devicearray.DeviceNDArray,
    temp_buffer: cuda.devicearray.DeviceNDArray,
    qubit_count: int,
    targets: tuple,
    controls: tuple,
    control_state: tuple
) -> bool:
    """Apply single operation on GPU and return if buffers need swapping."""
    gate_type = getattr(op, "gate_type", None)
    
    if len(targets) == 1 and not controls:
        return _apply_single_qubit_gpu(result_buffer, temp_buffer, op.matrix, targets[0], gate_type)
    
    elif len(targets) == 2 and not controls:
        if gate_type == "cx":
            return _apply_cnot_gpu(result_buffer, temp_buffer, targets[0], targets[1])
        elif gate_type == "swap":
            return _apply_swap_gpu(result_buffer, temp_buffer, targets[0], targets[1])
        else:
            return _apply_two_qubit_gpu(result_buffer, temp_buffer, op.matrix, targets[0], targets[1])
    
    elif controls:
        if len(targets) == 1 and len(controls) == 1 and gate_type == "cphaseshift":
            _apply_controlled_phase_shift_gpu_inplace(result_buffer, op.matrix[1, 1], controls, targets[0])
            return False
        else:
            return _apply_controlled_gpu(result_buffer, temp_buffer, op, qubit_count)
    
    return False


def _apply_single_qubit_gpu(
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    matrix: np.ndarray,
    target: int,
    gate_type: str = None
) -> bool:
    """Apply single qubit gate on GPU using efficient kernels."""
    if gate_type and gate_type in DIAGONAL_GATES:
        _apply_diagonal_gate_gpu_inplace(state_gpu, matrix, target, out_gpu)
        return True
    else:
        _apply_single_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix, target, gate_type)
        return True


def _apply_cnot_gpu(
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    control: int,
    target: int
) -> bool:
    """Apply CNOT gate on GPU using efficient kernel."""
    _apply_cnot_gpu_inplace(state_gpu, control, target, out_gpu)
    return True


def _apply_swap_gpu(
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    qubit_0: int,
    qubit_1: int
) -> bool:
    """Apply SWAP gate on GPU using efficient kernel."""
    _apply_swap_gpu_inplace(state_gpu, qubit_0, qubit_1, out_gpu)
    return True


def _apply_two_qubit_gpu(
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    matrix: np.ndarray,
    target0: int,
    target1: int
) -> bool:
    """Apply two-qubit gate on GPU using efficient kernel."""
    _apply_two_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix, target0, target1)
    return True


def _apply_controlled_gpu(
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    op: GateOperation,
    qubit_count: int
) -> bool:
    """Apply controlled gate on GPU using efficient kernel with optimal configuration."""
    targets = op.targets
    num_ctrl = len(getattr(op, "_ctrl_modifiers", []))
    matrix = op.matrix
    
    total_size = state_gpu.size
    
    control_mask = 0
    control_state_mask = 0
    ctrl_modifiers = getattr(op, "_ctrl_modifiers", [])
    for ctrl, state_val in zip(targets[:num_ctrl], ctrl_modifiers):
        bit_pos = qubit_count - 1 - ctrl
        control_mask |= 1 << bit_pos
        if state_val == 1:
            control_state_mask |= 1 << bit_pos
    
    target_mask = 0
    for target in targets[num_ctrl:]:
        target_mask |= 1 << (qubit_count - 1 - target)
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid, threads_per_block = _get_optimal_launch_config(total_size, qubit_count)
    
    matrix_size = matrix.shape[0]
    if matrix_size <= 16 and qubit_count >= 20:
        matrix_gpu = cuda.to_device(np.ascontiguousarray(matrix.flatten()))
    else:
        matrix_gpu = cuda.to_device(matrix.flatten())
    
    _controlled_gate_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, total_size, matrix_size
    )
    
    return True


@cuda.jit(inline=True, fastmath=True)
def _controlled_gate_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                           control_state_mask, n_qubits, total_size, matrix_size):
    """Controlled gate kernel using efficient bit manipulation and grid-stride loops."""
    i_start = cuda.grid(1)
    threads_per_grid = cuda.gridsize(1)
    
    for i in range(i_start, total_size, threads_per_grid):
        if (i & control_mask) == control_state_mask:
            target_state = 0
            for bit in range(matrix_size):
                if i & (target_mask >> bit):
                    target_state |= (1 << bit)
            
            new_amplitude = 0j
            for j in range(matrix_size):
                matrix_element = matrix_flat[target_state * matrix_size + j]
                
                target_idx = i & ~target_mask
                for bit in range(matrix_size):
                    if j & (1 << bit):
                        target_idx |= (target_mask >> (matrix_size - 1 - bit))
                
                new_amplitude += matrix_element * state_flat[target_idx]
            
            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]
