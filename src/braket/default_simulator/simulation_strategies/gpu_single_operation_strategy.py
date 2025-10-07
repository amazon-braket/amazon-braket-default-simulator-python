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

This module provides GPU acceleration for quantum circuit execution with
persistent state management and warp-cooperative kernels.
"""

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
    """GPU quantum operations with single write/read pattern."""
    use_gpu = _GPU_AVAILABLE and qubit_count >= 8 and state.size >= 256
    
    if not use_gpu:
        print(f"Using CPU for {qubit_count} qubits ({state.size} elements)")
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    print(f"Using GPU for {qubit_count} qubits ({state.size} elements)")
    
    return _execute_gpu_path(state, qubit_count, operations)


def clear_gpu_caches():
    """Clear all GPU caches and reset state."""
    gpu_manager = get_persistent_gpu_manager()
    if gpu_manager is not None:
        gpu_manager.clear_cache()


def _execute_gpu_path(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """GPU execution: write once to GPU, run all operations on GPU, write back to CPU once."""
    gpu_manager = get_persistent_gpu_manager()
    
    if gpu_manager is not None and qubit_count >= 22:
        current_buffer, output_buffer = gpu_manager.get_persistent_state(state, force_refresh=False)
        
        if _tensor_core_accelerator and len(operations) > 4:
            accelerated_ops, accel_info = accelerate_with_tensor_cores(operations)
            if accel_info.get('accelerated', False):
                print(f"Tensor core acceleration: {accel_info['operations_accelerated']} operations")
                operations = accelerated_ops
        
        if _warp_optimizer and state.size >= 1048576:
            _execute_warp_cooperative(current_buffer, output_buffer, operations, qubit_count)
            return gpu_manager.get_result_array(output_buffer, use_zero_copy=True)
        
        _execute_operations_on_gpu(current_buffer, output_buffer, operations, qubit_count)
        return gpu_manager.get_result_array(output_buffer, use_zero_copy=True)
    
    else:
        stream = cuda.stream()
        current_buffer = cuda.to_device(state, stream=stream)
        output_buffer = cuda.device_array_like(current_buffer)
        stream.synchronize()
        
        _execute_operations_on_gpu(current_buffer, output_buffer, operations, qubit_count)
        
        result = output_buffer.copy_to_host(stream=stream)
        stream.synchronize()
        return result


def _execute_warp_cooperative(
    current_buffer, output_buffer, operations: list[GateOperation], qubit_count: int
):
    """Execute operations using warp-cooperative kernels."""
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
            elif len(getattr(op, "_ctrl_modifiers", [])) > 0:
                apply_controlled_warp_cooperative(current_buffer, output_buffer, op, qubit_count, {})
            else:
                _apply_operation_gpu(op, current_buffer, output_buffer, qubit_count)
            
            current_buffer, output_buffer = output_buffer, current_buffer


def _execute_operations_on_gpu(
    current_buffer, output_buffer, operations: list[GateOperation], qubit_count: int
):
    """Execute all operations on GPU with ping-pong buffers."""
    for op in operations:
        _apply_operation_gpu(op, current_buffer, output_buffer, qubit_count)
        current_buffer, output_buffer = output_buffer, current_buffer


def _group_fusable_operations(operations: list[GateOperation]) -> list[list[GateOperation]]:
    """Group operations that can be fused together."""
    sequences = []
    current_sequence = []
    
    for op in operations:
        if (len(op.targets) <= 2 and 
            len(getattr(op, "_ctrl_modifiers", [])) == 0 and 
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


def _apply_operation_gpu(
    op: GateOperation, 
    input_buffer: cuda.devicearray.DeviceNDArray, 
    output_buffer: cuda.devicearray.DeviceNDArray, 
    qubit_count: int
):
    """Apply single operation on GPU with appropriate kernel."""
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
            _apply_diagonal_gate_warp(input_buffer, op.matrix, targets[0], output_buffer, threads_per_block)
        else:
            _apply_single_qubit_gate_warp(input_buffer, output_buffer, op.matrix, targets[0], threads_per_block)
    
    elif len(targets) == 2:
        if gate_type == "cx":
            _apply_cnot_warp(input_buffer, targets[0], targets[1], output_buffer, threads_per_block)
        elif gate_type == "swap":
            _apply_swap_warp(input_buffer, targets[0], targets[1], output_buffer, threads_per_block)
        else:
            _apply_two_qubit_gate_warp(input_buffer, output_buffer, op.matrix, targets[0], targets[1], threads_per_block)
    
    elif len(getattr(op, "_ctrl_modifiers", [])) > 0:
        _apply_controlled_gate_warp(input_buffer, output_buffer, op, qubit_count, threads_per_block)


def _apply_diagonal_gate_warp(state_gpu, matrix, target, out_gpu, threads_per_block):
    """Warp-cooperative diagonal gate implementation."""
    a, d = matrix[0, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    total_size = state_gpu.size
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (total_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _warp_diagonal_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a.real, a.imag, d.real, d.imag, target_mask, total_size
    )


def _apply_single_qubit_gate_warp(state_gpu, out_gpu, matrix, target, threads_per_block):
    """Warp-cooperative single qubit gate implementation."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    half_size = state_gpu.size >> 1
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (half_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _warp_single_qubit_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size
    )


def _apply_cnot_warp(state_gpu, control, target, out_gpu, threads_per_block):
    """Warp-cooperative CNOT gate implementation."""
    n_qubits = len(state_gpu.shape)
    control_bit = n_qubits - control - 1
    target_bit = n_qubits - target - 1
    control_mask = 1 << control_bit
    target_mask = 1 << target_bit
    
    quarter_size = state_gpu.size >> 2
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    blocks_per_grid = min(
        (quarter_size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _warp_cnot_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, control_mask, target_mask, quarter_size
    )


def _apply_swap_warp(state_gpu, qubit_0, qubit_1, out_gpu, threads_per_block):
    """Warp-cooperative SWAP gate implementation."""
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
    
    _warp_swap_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, iterations
    )


def _apply_two_qubit_gate_warp(state_gpu, out_gpu, matrix, target0, target1, threads_per_block):
    """Warp-cooperative two-qubit gate implementation."""
    n_qubits = len(state_gpu.shape)
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
        (state_gpu.size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _warp_two_qubit_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, state_gpu.size
    )


def _apply_controlled_gate_warp(state_gpu, out_gpu, op, qubit_count, threads_per_block):
    """Warp-cooperative controlled gate implementation."""
    targets = op.targets
    num_ctrl = len(getattr(op, "_ctrl_modifiers", []))
    matrix = op.matrix
    
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
    
    matrix_size = matrix.shape[0]
    matrix_gpu = cuda.to_device(matrix.flatten())
    
    blocks_per_grid = min(
        (state_gpu.size + threads_per_block - 1) // threads_per_block, 
        _MAX_BLOCKS_PER_GRID
    )
    
    _warp_controlled_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, state_gpu.size, matrix_size
    )


@cuda.jit(inline=True, fastmath=True)
def _warp_diagonal_kernel(state_flat, out_flat, a_real, a_imag, d_real, d_imag, target_mask, total_size):
    """Warp-cooperative diagonal gate with perfect coalescing."""
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    global_warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
    
    elements_per_warp = 32
    warp_base = global_warp_id * elements_per_warp
    
    if warp_base + lane_id < total_size:
        i = warp_base + lane_id
        
        mask_bit = i & target_mask
        if mask_bit:
            factor_real = d_real
            factor_imag = d_imag
        else:
            factor_real = a_real
            factor_imag = a_imag
        
        state_value = state_flat[i]
        result = complex(
            factor_real * state_value.real - factor_imag * state_value.imag,
            factor_real * state_value.imag + factor_imag * state_value.real
        )
        
        out_flat[i] = result


@cuda.jit(inline=True, fastmath=True)
def _warp_single_qubit_kernel(state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size):
    """Warp-cooperative single qubit gate with 32-thread coordination."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    lane_id = cuda.threadIdx.x % 32
    
    while i < half_size:
        idx0 = i + (i & ~(target_mask - 1))
        idx1 = idx0 | target_mask
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        shuffled_s0_real = cuda.shfl_sync(0xFFFFFFFF, s0.real, lane_id)
        shuffled_s0_imag = cuda.shfl_sync(0xFFFFFFFF, s0.imag, lane_id)
        shuffled_s1_real = cuda.shfl_sync(0xFFFFFFFF, s1.real, lane_id)
        shuffled_s1_imag = cuda.shfl_sync(0xFFFFFFFF, s1.imag, lane_id)
        
        shuffled_s0 = complex(shuffled_s0_real, shuffled_s0_imag)
        shuffled_s1 = complex(shuffled_s1_real, shuffled_s1_imag)
        
        result0 = a * shuffled_s0 + b * shuffled_s1
        result1 = c * shuffled_s0 + d * shuffled_s1
        
        out_flat[idx0] = result0
        out_flat[idx1] = result1
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _warp_cnot_kernel(state_flat, out_flat, control_mask, target_mask, quarter_size):
    """Warp-cooperative CNOT gate with shuffle-based data exchange."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    lane_id = cuda.threadIdx.x % 32
    
    while i < quarter_size:
        control_idx = i | control_mask
        target_idx = control_idx | target_mask
        
        control_state = state_flat[control_idx]
        target_state = state_flat[target_idx]
        
        control_real = cuda.shfl_sync(0xFFFFFFFF, control_state.real, lane_id)
        control_imag = cuda.shfl_sync(0xFFFFFFFF, control_state.imag, lane_id)
        target_real = cuda.shfl_sync(0xFFFFFFFF, target_state.real, lane_id)
        target_imag = cuda.shfl_sync(0xFFFFFFFF, target_state.imag, lane_id)
        
        cooperative_control = complex(control_real, control_imag)
        cooperative_target = complex(target_real, target_imag)
        
        out_flat[control_idx] = cooperative_target
        out_flat[target_idx] = cooperative_control
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _warp_swap_kernel(state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, iterations):
    """Warp-cooperative SWAP kernel."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    lane_id = cuda.threadIdx.x % 32
    
    while i < iterations:
        base = i + ((i >> pos_0) << pos_0)
        base += (base >> pos_1) << pos_1
        
        idx0 = base | mask_1
        idx1 = base | mask_0
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        cooperative_s0_real = cuda.shfl_sync(0xFFFFFFFF, s0.real, lane_id)
        cooperative_s0_imag = cuda.shfl_sync(0xFFFFFFFF, s0.imag, lane_id)
        cooperative_s1_real = cuda.shfl_sync(0xFFFFFFFF, s1.real, lane_id)
        cooperative_s1_imag = cuda.shfl_sync(0xFFFFFFFF, s1.imag, lane_id)
        
        out_flat[idx0] = complex(cooperative_s1_real, cooperative_s1_imag)
        out_flat[idx1] = complex(cooperative_s0_real, cooperative_s0_imag)
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _warp_two_qubit_kernel(state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
                          m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, total_size):
    """Warp-cooperative two qubit gate with component-wise shuffle operations."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    lane_id = cuda.threadIdx.x % 32
    
    while i < total_size:
        if (i & mask_both) == 0:
            base_idx = i
            idx1 = base_idx | mask_1
            idx2 = base_idx | mask_0  
            idx3 = base_idx | mask_both
            
            s0 = state_flat[base_idx]
            s1 = state_flat[idx1]
            s2 = state_flat[idx2]
            s3 = state_flat[idx3]
            
            s0_real = cuda.shfl_sync(0xFFFFFFFF, s0.real, lane_id)
            s0_imag = cuda.shfl_sync(0xFFFFFFFF, s0.imag, lane_id)
            s1_real = cuda.shfl_sync(0xFFFFFFFF, s1.real, lane_id)
            s1_imag = cuda.shfl_sync(0xFFFFFFFF, s1.imag, lane_id)
            s2_real = cuda.shfl_sync(0xFFFFFFFF, s2.real, lane_id)
            s2_imag = cuda.shfl_sync(0xFFFFFFFF, s2.imag, lane_id)
            s3_real = cuda.shfl_sync(0xFFFFFFFF, s3.real, lane_id)
            s3_imag = cuda.shfl_sync(0xFFFFFFFF, s3.imag, lane_id)
            
            cooperative_s0 = complex(s0_real, s0_imag)
            cooperative_s1 = complex(s1_real, s1_imag)
            cooperative_s2 = complex(s2_real, s2_imag)
            cooperative_s3 = complex(s3_real, s3_imag)
            
            r0 = m00 * cooperative_s0 + m01 * cooperative_s1 + m02 * cooperative_s2 + m03 * cooperative_s3
            r1 = m10 * cooperative_s0 + m11 * cooperative_s1 + m12 * cooperative_s2 + m13 * cooperative_s3
            r2 = m20 * cooperative_s0 + m21 * cooperative_s1 + m22 * cooperative_s2 + m23 * cooperative_s3
            r3 = m30 * cooperative_s0 + m31 * cooperative_s1 + m32 * cooperative_s2 + m33 * cooperative_s3
            
            out_flat[base_idx] = r0
            out_flat[idx1] = r1
            out_flat[idx2] = r2
            out_flat[idx3] = r3
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _warp_controlled_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                           control_state_mask, n_qubits, total_size, matrix_size):
    """Warp-cooperative controlled gate with efficient mask operations."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    lane_id = cuda.threadIdx.x % 32
    
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
                
                state_value = state_flat[target_idx]
                
                cooperative_real = cuda.shfl_sync(0xFFFFFFFF, state_value.real, lane_id)
                cooperative_imag = cuda.shfl_sync(0xFFFFFFFF, state_value.imag, lane_id)
                cooperative_value = complex(cooperative_real, cooperative_imag)
                
                new_amplitude += matrix_element * cooperative_value
            
            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]
        
        i += stride
