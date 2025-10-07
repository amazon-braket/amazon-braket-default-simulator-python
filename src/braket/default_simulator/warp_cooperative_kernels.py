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
Warp-level cooperative kernel implementations for quantum operations.

This module provides highly optimized warp-cooperative CUDA kernels that leverage
32-thread warp synchronization, shuffle operations, and cooperative execution
patterns for maximum GPU efficiency in quantum state manipulation.
"""

from typing import Dict, List, Optional

import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
)


class WarpCooperativeOptimizer:
    """Warp-level cooperative quantum operations for maximum GPU efficiency."""
    
    def __init__(self):
        self.warp_size = 32
        self.max_warps_per_block = 32
        self.shared_memory_banks = 32
        self.device_properties = self._get_device_properties()
    
    def _get_device_properties(self) -> dict:
        """Get device properties for warp-level optimization."""
        if not _GPU_AVAILABLE:
            return {}
        
        device = cuda.get_current_device()
        return {
            'compute_capability': device.compute_capability,
            'max_shared_memory': device.MAX_SHARED_MEMORY_PER_BLOCK,
            'multiprocessor_count': device.MULTIPROCESSOR_COUNT,
            'warp_scheduler_count': 4
        }
    
    def calculate_optimal_warp_configuration(self, state_size: int, operation_complexity: int) -> dict:
        """Calculate optimal warp configuration for cooperative operations."""
        optimal_warps = min(
            self.max_warps_per_block,
            max(1, state_size // (self.warp_size * 32))
        )
        
        threads_per_block = optimal_warps * self.warp_size
        blocks_per_grid = min(
            (state_size + threads_per_block - 1) // threads_per_block,
            self.device_properties.get('multiprocessor_count', 80) * 2
        )
        
        return {
            'warps_per_block': optimal_warps,
            'threads_per_block': threads_per_block,
            'blocks_per_grid': blocks_per_grid,
            'shared_memory_per_warp': self.device_properties.get('max_shared_memory', 49152) // optimal_warps,
            'cooperative_threads': self.warp_size
        }


_warp_optimizer = WarpCooperativeOptimizer() if _GPU_AVAILABLE else None


@cuda.jit(inline=True, fastmath=True)
def _warp_cooperative_single_qubit_kernel(state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size):
    """Warp-cooperative single qubit gate with optimized 32-thread coordination."""
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
def _warp_cooperative_diagonal_kernel(state_flat, out_flat, a_real, a_imag, d_real, d_imag, target_mask, total_size):
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
def _warp_cooperative_cnot_kernel(state_flat, out_flat, control_mask, target_mask, quarter_size):
    """Optimized warp-cooperative CNOT gate with shuffle-based data exchange."""
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
def _warp_cooperative_two_qubit_kernel(state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
                                      m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, total_size):
    """Optimized warp-cooperative two qubit gate with component-wise shuffle operations."""
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
def _warp_cooperative_controlled_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                                       control_state_mask, n_qubits, total_size, matrix_size):
    """Optimized warp-cooperative controlled gate with efficient mask operations."""
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


@cuda.jit(inline=True, fastmath=True)
def _warp_cooperative_probability_reduction_kernel(state_flat, probabilities, target_qubits_mask, total_size):
    """Warp-cooperative probability calculation with reduction operations."""
    warp_reduction = cuda.shared.array(1024, cuda.float64)
    
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    global_warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
    
    elements_per_warp = 32
    warp_base = global_warp_id * elements_per_warp
    
    local_sum = 0.0
    
    if warp_base + lane_id < total_size:
        i = warp_base + lane_id
        
        if (i & target_qubits_mask) == target_qubits_mask:
            amplitude = state_flat[i]
            local_probability = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
            local_sum = local_probability
    
    for offset in [16, 8, 4, 2, 1]:
        other_sum = cuda.shfl_down_sync(0xFFFFFFFF, local_sum, offset)
        if lane_id < offset:
            local_sum += other_sum
    
    if lane_id == 0:
        warp_reduction[warp_id] = local_sum
    
    cuda.syncthreads()
    
    if cuda.threadIdx.x < 32:
        warp_sum = warp_reduction[cuda.threadIdx.x] if cuda.threadIdx.x < (cuda.blockDim.x + 31) // 32 else 0.0
        
        for offset in [16, 8, 4, 2, 1]:
            other_sum = cuda.shfl_down_sync(0xFFFFFFFF, warp_sum, offset)
            if cuda.threadIdx.x < offset:
                warp_sum += other_sum
        
        if cuda.threadIdx.x == 0:
            cuda.atomic.add(probabilities, 0, warp_sum)


@cuda.jit(inline=True, fastmath=True)
def _warp_cooperative_fused_sequence_kernel(state_flat, out_flat, gate_data, num_gates, n_qubits, total_size):
    """Optimized warp-cooperative fused gate sequence with shuffle operations."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    lane_id = cuda.threadIdx.x % 32
    
    while i < total_size:
        amplitude = state_flat[i]
        
        for gate_idx in range(num_gates):
            gate_type = int(gate_data[gate_idx, 0].real)
            target = int(gate_data[gate_idx, 1].real)
            
            if gate_type == 1:
                target_bit = n_qubits - target - 1
                paired_idx = i ^ (1 << target_bit)
                
                paired_amplitude = state_flat[paired_idx]
                
                cooperative_real = cuda.shfl_sync(0xFFFFFFFF, amplitude.real, lane_id)
                cooperative_imag = cuda.shfl_sync(0xFFFFFFFF, amplitude.imag, lane_id)
                cooperative_amplitude = complex(cooperative_real, cooperative_imag)
                
                if i <= paired_idx:
                    temp = amplitude
                    amplitude = paired_amplitude
                    if i != paired_idx:
                        out_flat[paired_idx] = temp
            
            elif gate_type == 3:
                target_bit = n_qubits - target - 1
                if (i >> target_bit) & 1:
                    amplitude *= -1
            
            elif gate_type == 4:
                target_bit = n_qubits - target - 1
                paired_idx = i ^ (1 << target_bit)
                paired_amplitude = state_flat[paired_idx]
                
                inv_sqrt2 = 0.7071067811865476
                if i <= paired_idx:
                    new_amplitude = inv_sqrt2 * (amplitude + paired_amplitude)
                    paired_new_amplitude = inv_sqrt2 * (amplitude - paired_amplitude)
                    amplitude = new_amplitude
                    if i != paired_idx:
                        out_flat[paired_idx] = paired_new_amplitude
            
            elif gate_type == 5:
                control = int(gate_data[gate_idx, 2].real)
                control_bit = n_qubits - control - 1
                target_bit = n_qubits - target - 1
                
                if (i >> control_bit) & 1:
                    swap_idx = i ^ (1 << target_bit)
                    swap_amplitude = state_flat[swap_idx]
                    
                    cooperative_real = cuda.shfl_sync(0xFFFFFFFF, amplitude.real, lane_id)
                    cooperative_imag = cuda.shfl_sync(0xFFFFFFFF, amplitude.imag, lane_id)
                    
                    amplitude = swap_amplitude
                    out_flat[swap_idx] = complex(cooperative_real, cooperative_imag)
        
        out_flat[i] = amplitude


def apply_single_qubit_warp_cooperative(state_gpu, out_gpu, matrix, target, qubit_count):
    """Apply single qubit gate using warp-cooperative optimization."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = qubit_count
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    half_size = state_gpu.size >> 1
    
    warp_config = _warp_optimizer.calculate_optimal_warp_configuration(half_size, 1)
    
    _warp_cooperative_single_qubit_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
        state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size
    )


def apply_diagonal_warp_cooperative(state_gpu, matrix, target, out_gpu):
    """Apply diagonal gate using warp-cooperative optimization."""
    a, d = matrix[0, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    warp_config = _warp_optimizer.calculate_optimal_warp_configuration(state_gpu.size, 1)
    
    _warp_cooperative_diagonal_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
        state_flat, out_flat, a.real, a.imag, d.real, d.imag, target_mask, state_gpu.size
    )


def apply_cnot_warp_cooperative(state_gpu, control, target, out_gpu, qubit_count):
    """Apply CNOT gate using warp-cooperative optimization."""
    n_qubits = qubit_count
    control_bit = n_qubits - control - 1
    target_bit = n_qubits - target - 1
    control_mask = 1 << control_bit
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    quarter_size = state_gpu.size >> 2
    
    warp_config = _warp_optimizer.calculate_optimal_warp_configuration(quarter_size, 2)
    
    _warp_cooperative_cnot_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
        state_flat, out_flat, control_mask, target_mask, quarter_size
    )


def apply_two_qubit_warp_cooperative(state_gpu, out_gpu, matrix, target0, target1, qubit_count):
    """Apply two qubit gate using warp-cooperative optimization."""
    n_qubits = qubit_count
    mask_0 = 1 << (n_qubits - 1 - target0)
    mask_1 = 1 << (n_qubits - 1 - target1)
    mask_both = mask_0 | mask_1
    
    m00, m01, m02, m03 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    m10, m11, m12, m13 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    m20, m21, m22, m23 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    m30, m31, m32, m33 = matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    warp_config = _warp_optimizer.calculate_optimal_warp_configuration(state_gpu.size, 4)
    
    _warp_cooperative_two_qubit_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
        state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, state_gpu.size
    )


def apply_controlled_warp_cooperative(state_gpu, out_gpu, op, qubit_count, matrix_cache):
    """Apply controlled gate using warp-cooperative optimization."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    matrix = op.matrix
    
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
    cache_key = f"warp_ctrl_{matrix_size}_{hash(matrix.tobytes())}"
    
    if matrix_cache and cache_key in matrix_cache:
        matrix_gpu = matrix_cache[cache_key]
    else:
        matrix_contiguous = np.ascontiguousarray(matrix.flatten())
        matrix_gpu = cuda.to_device(matrix_contiguous)
        if matrix_cache:
            matrix_cache[cache_key] = matrix_gpu
    
    warp_config = _warp_optimizer.calculate_optimal_warp_configuration(state_gpu.size, matrix_size)
    
    _warp_cooperative_controlled_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, state_gpu.size, matrix_size
    )


def execute_warp_cooperative_fused_sequence(state_gpu, out_gpu, gate_data_gpu, num_gates, qubit_count):
    """Execute fused gate sequence using warp-cooperative optimization."""
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    warp_config = _warp_optimizer.calculate_optimal_warp_configuration(state_gpu.size, num_gates)
    
    _warp_cooperative_fused_sequence_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
        state_flat, out_flat, gate_data_gpu, num_gates, qubit_count, state_gpu.size
    )


def calculate_measurement_probabilities_warp_cooperative(state_gpu, target_qubits) -> np.ndarray:
    """Calculate measurement probabilities using warp-cooperative reductions."""
    if not _warp_optimizer:
        return np.array([0.0])
    
    state_flat = state_gpu.reshape(-1)
    
    num_outcomes = 2 ** len(target_qubits)
    probabilities_gpu = cuda.device_array(num_outcomes, dtype=np.float64)
    
    for outcome in range(num_outcomes):
        target_mask = 0
        for i, qubit in enumerate(target_qubits):
            if outcome & (1 << i):
                target_mask |= 1 << (len(state_gpu.shape) - 1 - qubit)
        
        prob_result = cuda.device_array(1, dtype=np.float64)
        
        warp_config = _warp_optimizer.calculate_optimal_warp_configuration(state_gpu.size, 1)
        
        _warp_cooperative_probability_reduction_kernel[warp_config['blocks_per_grid'], warp_config['threads_per_block']](
            state_flat, prob_result, target_mask, state_gpu.size
        )
        
        probabilities_gpu[outcome] = prob_result[0]
    
    return probabilities_gpu.copy_to_host()
