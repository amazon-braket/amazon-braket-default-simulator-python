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
GPU-optimized single operation strategy for quantum simulations.

This module provides a highly optimized execution strategy that leverages GPU
acceleration with persistent state management, kernel fusion, and advanced
memory optimization for maximum quantum simulation performance.
"""

from typing import Dict, List, Tuple
import time

import numpy as np
from numba import cuda

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

from braket.default_simulator.circuit_compiler import (
    execute_template_fused_kernel,
    _circuit_compiler,
)
from braket.default_simulator.mega_kernel_generator import (
    execute_mega_kernel_circuit,
    _mega_kernel_generator,
)
from braket.default_simulator.tensor_core_acceleration import (
    accelerate_with_tensor_cores,
    _tensor_core_accelerator,
)
from braket.default_simulator.warp_cooperative_kernels import (
    apply_cnot_warp_cooperative,
    apply_controlled_warp_cooperative,
    apply_diagonal_warp_cooperative,
    apply_single_qubit_warp_cooperative,
    apply_two_qubit_warp_cooperative,
    execute_warp_cooperative_fused_sequence,
    _warp_optimizer,
)


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Optimized quantum operations with intelligent CPU/GPU routing and advanced acceleration."""
    use_gpu = _GPU_AVAILABLE and qubit_count >= 8 and state.size >= 256
    
    if not use_gpu:
        print(f"Using CPU for {qubit_count} qubits ({state.size} elements)")
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    print(f"Using GPU for {qubit_count} qubits ({state.size} elements)")
    
    if qubit_count >= 22:
        return _execute_high_performance_path(state, qubit_count, operations)
    else:
        return _execute_optimized_fast_path(state, qubit_count, operations)


def clear_gpu_caches():
    """Clear all GPU caches and reset state."""
    gpu_manager = get_persistent_gpu_manager()
    if gpu_manager is not None:
        gpu_manager.clear_cache()


def _execute_high_performance_path(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """High-performance GPU execution for large qubit counts using advanced optimizations."""
    gpu_manager = get_persistent_gpu_manager()
    
    if gpu_manager is None:
        return _execute_optimized_fast_path(state, qubit_count, operations)
    
    current_buffer, output_buffer = gpu_manager.get_persistent_state(state, force_refresh=False)
    
    if _tensor_core_accelerator and len(operations) > 4:
        accelerated_ops, accel_info = accelerate_with_tensor_cores(operations)
        if accel_info.get('accelerated', False):
            print(f"Tensor core acceleration: {accel_info['operations_accelerated']} operations")
            operations = accelerated_ops
    
    if _circuit_compiler and len(operations) > 8:
        compiled_kernel = _circuit_compiler.compile_circuit(operations, qubit_count)
        if compiled_kernel:
            execute_template_fused_kernel(current_buffer, compiled_kernel, qubit_count)
            return gpu_manager.get_result_array(current_buffer, use_zero_copy=True)
    
    if _mega_kernel_generator and len(operations) > 16:
        mega_kernel_success = execute_mega_kernel_circuit(
            current_buffer, output_buffer, operations, qubit_count
        )
        if mega_kernel_success:
            return gpu_manager.get_result_array(output_buffer, use_zero_copy=True)
    
    if _warp_optimizer and state.size >= 1048576:
        return _execute_warp_cooperative_path(
            current_buffer, output_buffer, operations, qubit_count, gpu_manager
        )
    
    for i, op in enumerate(operations):
        _apply_operation_advanced_dispatch(
            op, current_buffer, output_buffer, qubit_count, i == len(operations) - 1
        )
        current_buffer, output_buffer = output_buffer, current_buffer
    
    return gpu_manager.get_result_array(current_buffer, use_zero_copy=True)


def _execute_optimized_fast_path(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Optimized GPU execution with minimal memory transfers and better utilization."""
    stream = cuda.stream()
    
    current_buffer = cuda.to_device(state, stream=stream)
    output_buffer = cuda.device_array_like(current_buffer)
    
    stream.synchronize()
    
    for op in operations:
        _apply_operation_direct_dispatch(op, current_buffer, output_buffer, qubit_count, current_buffer.size)
        current_buffer, output_buffer = output_buffer, current_buffer
    
    result = current_buffer.copy_to_host(stream=stream)
    stream.synchronize()
    
    return result


def _execute_warp_cooperative_path(
    current_buffer, output_buffer, operations: list[GateOperation], 
    qubit_count: int, gpu_manager
) -> np.ndarray:
    """Execute operations using warp-cooperative kernels for maximum GPU efficiency."""
    
    fusable_sequences = _group_fusable_operations(operations)
    
    for sequence in fusable_sequences:
        if len(sequence) > 1:
            gate_data = _prepare_fused_gate_data(sequence, qubit_count)
            gate_data_gpu = cuda.to_device(gate_data)
            
            execute_warp_cooperative_fused_sequence(
                current_buffer, output_buffer, gate_data_gpu, len(sequence), qubit_count
            )
            current_buffer, output_buffer = output_buffer, current_buffer
        else:
            op = sequence[0]
            targets = op.targets
            gate_type = getattr(op, "gate_type", None)
            
            if len(targets) == 1:
                if gate_type in DIAGONAL_GATES:
                    apply_diagonal_warp_cooperative(current_buffer, op.matrix, targets[0], output_buffer)
                else:
                    apply_single_qubit_warp_cooperative(current_buffer, output_buffer, op.matrix, targets[0], qubit_count)
            elif len(targets) == 2:
                if gate_type == "cx":
                    apply_cnot_warp_cooperative(current_buffer, targets[0], targets[1], output_buffer, qubit_count)
                else:
                    apply_two_qubit_warp_cooperative(current_buffer, output_buffer, op.matrix, targets[0], targets[1], qubit_count)
            elif len(op._ctrl_modifiers) > 0:
                apply_controlled_warp_cooperative(current_buffer, output_buffer, op, qubit_count, {})
            else:
                _apply_operation_direct_dispatch(op, current_buffer, output_buffer, qubit_count, current_buffer.size)
            
            current_buffer, output_buffer = output_buffer, current_buffer
    
    return gpu_manager.get_result_array(current_buffer, use_zero_copy=True)


def _group_fusable_operations(operations: list[GateOperation]) -> list[list[GateOperation]]:
    """Group operations that can be fused together for better performance."""
    sequences = []
    current_sequence = []
    
    for op in operations:
        if (len(op.targets) <= 2 and 
            len(op._ctrl_modifiers) == 0 and 
            len(current_sequence) < 8):
            current_sequence.append(op)
        else:
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
            sequences.append([op])
    
    if current_sequence:
        sequences.append(current_sequence)
    
    return sequences


def _prepare_fused_gate_data(operations: list[GateOperation], qubit_count: int) -> np.ndarray:
    """Prepare gate data for fused execution."""
    gate_data = np.zeros((len(operations), 8), dtype=np.complex128)
    
    for i, op in enumerate(operations):
        targets = op.targets
        gate_type = getattr(op, "gate_type", None)
        
        if gate_type == "x":
            gate_data[i, 0] = 1
        elif gate_type == "y":
            gate_data[i, 0] = 2
        elif gate_type == "z":
            gate_data[i, 0] = 3
        elif gate_type == "h":
            gate_data[i, 0] = 4
        elif gate_type == "cx" and len(targets) == 2:
            gate_data[i, 0] = 5
            gate_data[i, 2] = targets[0]
        else:
            gate_data[i, 0] = 0
        
        gate_data[i, 1] = targets[0] if targets else 0
    
    return gate_data


def _apply_operation_advanced_dispatch(
    op: GateOperation, 
    input_buffer: cuda.devicearray.DeviceNDArray, 
    output_buffer: cuda.devicearray.DeviceNDArray, 
    qubit_count: int,
    is_final_op: bool
):
    """Advanced dispatch with optimized kernel selection based on operation characteristics."""
    targets = op.targets
    gate_type = getattr(op, "gate_type", None)
    state_size = input_buffer.size
    
    if state_size >= 4194304:
        threads_per_block = 1024
    elif state_size >= 1048576:
        threads_per_block = 512
    else:
        threads_per_block = 256
    
    if len(targets) == 1:
        if gate_type in DIAGONAL_GATES:
            _apply_diagonal_gate_gpu_optimized(input_buffer, op.matrix, targets[0], output_buffer, threads_per_block)
        else:
            _apply_single_qubit_gate_gpu_optimized(input_buffer, output_buffer, op.matrix, targets[0], threads_per_block)
    
    elif len(targets) == 2:
        if gate_type == "cx":
            _apply_cnot_gpu_optimized(input_buffer, targets[0], targets[1], output_buffer, threads_per_block)
        elif gate_type == "swap":
            _apply_swap_gpu_optimized(input_buffer, targets[0], targets[1], output_buffer, threads_per_block)
        else:
            _apply_two_qubit_gate_gpu_optimized(input_buffer, output_buffer, op.matrix, targets[0], targets[1], threads_per_block)
    
    elif len(op._ctrl_modifiers) > 0:
        _apply_controlled_gate_gpu_optimized(input_buffer, output_buffer, op, qubit_count, threads_per_block)


def _apply_operation_direct_dispatch(
    op: GateOperation, 
    input_buffer: cuda.devicearray.DeviceNDArray, 
    output_buffer: cuda.devicearray.DeviceNDArray, 
    qubit_count: int,
    state_size: int
):
    """Direct optimized dispatch using proven high-performance GPU kernels."""
    targets = op.targets
    gate_type = getattr(op, "gate_type", None)
    
    if len(targets) == 1:
        if gate_type == "pauli_z" or (gate_type in DIAGONAL_GATES):
            _apply_diagonal_gate_gpu_inplace(input_buffer, op.matrix, targets[0], output_buffer)
        else:
            _apply_single_qubit_gate_gpu_inplace(input_buffer, output_buffer, op.matrix, targets[0], gate_type)
    
    elif len(targets) == 2:
        if gate_type == "cx":
            _apply_cnot_gpu_inplace(input_buffer, targets[0], targets[1], output_buffer)
        elif gate_type == "swap":
            _apply_swap_gpu_inplace(input_buffer, targets[0], targets[1], output_buffer)
        else:
            _apply_two_qubit_gate_gpu_inplace(input_buffer, output_buffer, op.matrix, targets[0], targets[1])
    
    elif len(op._ctrl_modifiers) > 0:
        if len(targets) == 1 and len(op._ctrl_modifiers) == 1 and gate_type == "cphaseshift":
            _apply_controlled_phase_shift_gpu_inplace(input_buffer, op.matrix[1, 1], targets[:1], targets[1:][0])
        else:
            _apply_controlled_gate_gpu_direct(input_buffer, output_buffer, op, qubit_count)


def _apply_controlled_gate_gpu_direct(state_gpu, out_gpu, op: GateOperation, qubit_count: int):
    """Controlled gate implementation with standard optimization."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    matrix = op.matrix
    
    total_size = state_gpu.size
    
    control_mask = 0
    control_state_mask = 0
    for ctrl, state_val in zip(targets[:num_ctrl], op._ctrl_modifiers):
        bit_pos = qubit_count - 1 - ctrl
        control_mask |= 1 << bit_pos
        if state_val == 1:
            control_state_mask |= 1 << bit_pos
    
    target_mask = 0
    for target in targets[num_ctrl:]:
        target_mask |= 1 << (qubit_count - 1 - target)
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    matrix_size = matrix.shape[0]
    cache_key = f"ctrl_{matrix_size}_{hash(matrix.tobytes())}"
    matrix_gpu = cuda.to_device(matrix.flatten())
    
    threads_per_block = min(1024, max(512, total_size // 2048))
    blocks_per_grid = min(
        (total_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _controlled_gate_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, total_size, matrix_size
    )


@cuda.jit(inline=True, fastmath=True)
def _controlled_gate_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                           control_state_mask, n_qubits, total_size, matrix_size):
    """Standard controlled gate kernel for fallback operations."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
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
        
        i += stride

def _apply_diagonal_gate_gpu_optimized(state_gpu, matrix, target, out_gpu, threads_per_block):
    """Optimized diagonal gate implementation with configurable block size."""
    a, d = matrix[0, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    shifted_target_mask = target_mask - 1
    
    half_size = state_gpu.size >> 1
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (half_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _optimized_diagonal_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a, d, target_bit, target_mask, shifted_target_mask, half_size
    )


def _apply_single_qubit_gate_gpu_optimized(state_gpu, out_gpu, matrix, target, threads_per_block):
    """Optimized single qubit gate implementation with configurable block size."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    n = n_qubits - target - 1
    mask = (1 << n) - 1
    half_size = state_gpu.size >> 1
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (half_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _optimized_single_qubit_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a, b, c, d, n, mask, half_size
    )


def _apply_cnot_gpu_optimized(state_gpu, control, target, out_gpu, threads_per_block):
    """Optimized CNOT gate implementation with configurable block size."""
    n_qubits = len(state_gpu.shape)
    total_size = state_gpu.size
    
    control_bit = n_qubits - control - 1
    target_bit = n_qubits - target - 1
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (total_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _optimized_cnot_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, control_bit, target_bit, total_size
    )


def _apply_swap_gpu_optimized(state_gpu, qubit_0, qubit_1, out_gpu, threads_per_block):
    """Optimized SWAP gate implementation with configurable block size."""
    n_qubits = len(state_gpu.shape)
    iterations = state_gpu.size >> 2
    
    pos_0 = n_qubits - 1 - qubit_0
    pos_1 = n_qubits - 1 - qubit_1
    
    if pos_0 > pos_1:
        pos_0, pos_1 = pos_1, pos_0
    
    mask_0 = 1 << pos_0
    mask_1 = 1 << pos_1
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (iterations + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _optimized_swap_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, iterations
    )


def _apply_two_qubit_gate_gpu_optimized(state_gpu, out_gpu, matrix, target0, target1, threads_per_block):
    """Optimized two-qubit gate implementation with configurable block size."""
    n_qubits = len(state_gpu.shape)
    total_size = state_gpu.size
    
    mask_0 = 1 << (n_qubits - 1 - target0)
    mask_1 = 1 << (n_qubits - 1 - target1)
    mask_both = mask_0 | mask_1
    
    m00, m01, m02, m03 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    m10, m11, m12, m13 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    m20, m21, m22, m23 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    m30, m31, m32, m33 = matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (total_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _optimized_two_qubit_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, total_size
    )


def _apply_controlled_gate_gpu_optimized(state_gpu, out_gpu, op, qubit_count, threads_per_block):
    """Optimized controlled gate implementation with configurable block size."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    matrix = op.matrix
    
    total_size = state_gpu.size
    
    control_mask = 0
    control_state_mask = 0
    for ctrl, state_val in zip(targets[:num_ctrl], op._ctrl_modifiers):
        bit_pos = qubit_count - 1 - ctrl
        control_mask |= 1 << bit_pos
        if state_val == 1:
            control_state_mask |= 1 << bit_pos
    
    target_mask = 0
    for target in targets[num_ctrl:]:
        target_mask |= 1 << (qubit_count - 1 - target)
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    matrix_size = matrix.shape[0]
    matrix_gpu = cuda.to_device(matrix.flatten())
    
    blocks_per_grid = min(
        (total_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _optimized_controlled_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, total_size, matrix_size
    )


# High-performance CUDA kernels with better memory access patterns

@cuda.jit(inline=True, fastmath=True)
def _optimized_diagonal_kernel(state_flat, out_flat, a, d, target_bit, target_mask, shifted_target_mask, half_size):
    """Optimized diagonal gate kernel with improved memory coalescing."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < half_size:
        idx0 = (i & ~shifted_target_mask) << 1 | (i & shifted_target_mask)
        idx1 = idx0 | target_mask
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        out_flat[idx0] = a * s0
        out_flat[idx1] = d * s1
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_single_qubit_kernel(state_flat, out_flat, a, b, c, d, n, mask, half_size):
    """Optimized single qubit gate kernel with improved memory access."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < half_size:
        idx0 = ((i >> n) << (n + 1)) | (i & mask)
        idx1 = idx0 | (1 << n)
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        out_flat[idx0] = a * s0 + b * s1
        out_flat[idx1] = c * s0 + d * s1
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_cnot_kernel(state_flat, out_flat, control_bit, target_bit, total_size):
    """Optimized CNOT kernel with better branch divergence handling."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    control_mask = 1 << control_bit
    target_mask = 1 << target_bit
    
    while i < total_size:
        if (i & control_mask) != 0:
            partner_idx = i ^ target_mask
            out_flat[i] = state_flat[partner_idx]
        else:
            out_flat[i] = state_flat[i]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_swap_kernel(state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, iterations):
    """Optimized SWAP kernel with reduced divergence."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < iterations:
        base = i + ((i >> pos_0) << pos_0)
        base += (base >> pos_1) << pos_1
        
        idx0 = base | mask_1
        idx1 = base | mask_0
        
        out_flat[idx0] = state_flat[idx1]
        out_flat[idx1] = state_flat[idx0]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_two_qubit_kernel(state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13, 
                               m20, m21, m22, m23, m30, m31, m32, m33, 
                               mask_0, mask_1, mask_both, total_size):
    """Optimized two-qubit gate kernel with vectorized operations."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        if (i & mask_both) == 0:
            s0 = state_flat[i]
            s1 = state_flat[i | mask_1]
            s2 = state_flat[i | mask_0]
            s3 = state_flat[i | mask_both]
            
            out_flat[i] = m00 * s0 + m01 * s1 + m02 * s2 + m03 * s3
            out_flat[i | mask_1] = m10 * s0 + m11 * s1 + m12 * s2 + m13 * s3
            out_flat[i | mask_0] = m20 * s0 + m21 * s1 + m22 * s2 + m23 * s3
            out_flat[i | mask_both] = m30 * s0 + m31 * s1 + m32 * s2 + m33 * s3
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_controlled_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                                control_state_mask, n_qubits, total_size, matrix_size):
    """Optimized controlled gate kernel with reduced memory access."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
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
        
        i += stride
