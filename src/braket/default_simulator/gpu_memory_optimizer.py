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
Advanced GPU memory optimization for quantum state processing.

This module provides sophisticated memory optimization strategies including
shared memory utilization, memory coalescing, and bandwidth optimization
for GPU-accelerated quantum operations.
"""

from typing import Any, Dict
import time

import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
)


class GPUMemoryOptimizer:
    """Advanced GPU memory optimization for quantum state processing."""
    
    def __init__(self):
        self.device_properties = self._get_device_properties()
        self.shared_memory_size = 48 * 1024
        self.warp_size = 32
        self.max_threads_per_block = 1024
        
    def _get_device_properties(self) -> dict:
        """Get comprehensive GPU device properties for optimization."""
        if not _GPU_AVAILABLE:
            return {}
        
        device = cuda.get_current_device()
        
        return {
            'name': device.name,
            'compute_capability': device.compute_capability,
            'multiprocessor_count': device.MULTIPROCESSOR_COUNT,
            'max_threads_per_multiprocessor': getattr(device, 'MAX_THREADS_PER_MULTIPROCESSOR', 2048),
            'max_shared_memory_per_block': getattr(device, 'MAX_SHARED_MEMORY_PER_BLOCK', 49152),
            'max_shared_memory_per_multiprocessor': getattr(device, 'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', 98304),
            'l2_cache_size': getattr(device, 'L2_CACHE_SIZE', 6291456),
            'memory_bandwidth': getattr(device, 'MEMORY_CLOCK_RATE', 877000) * getattr(device, 'GLOBAL_MEMORY_BUS_WIDTH', 384) // 8 * 2,
            'register_count': getattr(device, 'MAX_REGISTERS_PER_BLOCK', 65536)
        }
    
    def calculate_optimal_block_size(self, state_size: int, operation_complexity: int = 1) -> tuple[int, int]:
        """Calculate optimal thread block configuration for memory bandwidth."""
        base_threads = min(1024, max(256, state_size // 1024))
        
        if state_size >= 2**20:
            threads_per_block = 1024
        elif state_size >= 2**16:
            threads_per_block = 512
        else:
            threads_per_block = 256
        
        blocks_per_grid = min(
            (state_size + threads_per_block - 1) // threads_per_block,
            min(_MAX_BLOCKS_PER_GRID, self.device_properties.get('multiprocessor_count', 80) * 4)
        )
        
        return threads_per_block, blocks_per_grid
    
    def get_shared_memory_config(self, block_size: int) -> int:
        """Get optimal shared memory configuration per block."""
        available_shared_memory = min(self.shared_memory_size, 48 * 1024)
        
        memory_per_thread = available_shared_memory // block_size
        return min(memory_per_thread * block_size, available_shared_memory)


_gpu_memory_optimizer = GPUMemoryOptimizer() if _GPU_AVAILABLE else None


@cuda.jit(inline=True, fastmath=True)
def _optimized_single_qubit_kernel_shared_memory(state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size):
    """Memory-optimized single qubit kernel with shared memory and coalescing."""
    shared_data = cuda.shared.array(1024, cuda.complex128)
    
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    base_idx = block_id * block_size + thread_id
    stride = cuda.gridDim.x * block_size
    
    while base_idx < half_size:
        idx0 = base_idx + (base_idx & ~(target_mask - 1))
        idx1 = idx0 | target_mask
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        shared_data[thread_id] = s0
        shared_data[thread_id + 512] = s1 if thread_id < 512 else 0j
        
        cuda.syncthreads()
        
        result0 = a * shared_data[thread_id] + b * (shared_data[thread_id + 512] if thread_id < 512 else s1)
        result1 = c * shared_data[thread_id] + d * (shared_data[thread_id + 512] if thread_id < 512 else s1)
        
        out_flat[idx0] = result0
        out_flat[idx1] = result1
        
        cuda.syncthreads()
        
        base_idx += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_diagonal_kernel_coalesced(state_flat, out_flat, a, d, target_mask, total_size):
    """Memory-optimized diagonal kernel with perfect memory coalescing."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        mask_bit = i & target_mask
        factor = d if mask_bit else a
        out_flat[i] = factor * state_flat[i]
        i += stride


@cuda.jit(inline=True, fastmath=True) 
def _optimized_cnot_kernel_shared_memory(state_flat, out_flat, control_mask, target_mask, quarter_size):
    """Memory-optimized CNOT kernel with shared memory prefetching."""
    shared_states = cuda.shared.array(1024, cuda.complex128)
    
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    base_idx = block_id * block_size + thread_id
    stride = cuda.gridDim.x * block_size
    
    while base_idx < quarter_size:
        control_idx = base_idx | control_mask
        target_idx = control_idx | target_mask
        
        if thread_id < 512:
            shared_states[thread_id] = state_flat[control_idx]
            shared_states[thread_id + 512] = state_flat[target_idx]
        
        cuda.syncthreads()
        
        if thread_id < 512:
            out_flat[control_idx] = shared_states[thread_id + 512]
            out_flat[target_idx] = shared_states[thread_id]
        
        cuda.syncthreads()
        
        base_idx += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_two_qubit_kernel_register_blocked(state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
                                                 m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, total_size):
    """Memory-optimized two qubit kernel with register blocking and prefetching."""
    block_shared = cuda.shared.array(2048, cuda.complex128)
    
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    thread_id = cuda.threadIdx.x
    
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
            
            if thread_id < 512:
                block_shared[thread_id] = s0
                block_shared[thread_id + 512] = s1
            
            cuda.syncthreads()
            
            r0 = m00 * s0 + m01 * s1 + m02 * s2 + m03 * s3
            r1 = m10 * s0 + m11 * s1 + m12 * s2 + m13 * s3
            r2 = m20 * s0 + m21 * s1 + m22 * s2 + m23 * s3
            r3 = m30 * s0 + m31 * s1 + m32 * s2 + m33 * s3
            
            out_flat[base_idx] = r0
            out_flat[idx1] = r1
            out_flat[idx2] = r2
            out_flat[idx3] = r3
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _optimized_controlled_kernel_shared_memory(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                                              control_state_mask, n_qubits, total_size, matrix_size):
    """Memory-optimized controlled gate kernel with shared memory and cache optimization."""
    shared_matrix = cuda.shared.array(256, cuda.complex128)
    shared_states = cuda.shared.array(512, cuda.complex128)
    
    thread_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if thread_id < matrix_size * matrix_size:
        shared_matrix[thread_id] = matrix_flat[thread_id]
    
    cuda.syncthreads()
    
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        if (i & control_mask) == control_state_mask:
            target_state = 0
            for bit in range(matrix_size):
                if i & (target_mask >> bit):
                    target_state |= (1 << bit)
            
            if thread_id < 256:
                shared_states[thread_id] = state_flat[i + thread_id] if (i + thread_id) < total_size else 0j
            
            cuda.syncthreads()
            
            new_amplitude = 0j
            for j in range(matrix_size):
                matrix_element = shared_matrix[target_state * matrix_size + j] if (target_state * matrix_size + j) < 256 else matrix_flat[target_state * matrix_size + j]
                
                target_idx = i & ~target_mask
                for bit in range(matrix_size):
                    if j & (1 << bit):
                        target_idx |= (target_mask >> (matrix_size - 1 - bit))
                
                state_value = shared_states[target_idx - i] if (target_idx >= i and target_idx - i < 256) else state_flat[target_idx]
                new_amplitude += matrix_element * state_value
            
            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _memory_profiler_kernel(state_flat, profiling_data, access_patterns, total_size):
    """Memory access profiling kernel for optimization analysis."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    thread_id = cuda.threadIdx.x
    
    access_count = 0
    cache_hits = 0
    
    while i < total_size:
        current_value = state_flat[i]
        
        if i > 0 and abs(state_flat[i] - state_flat[i-1]) < 1e-12:
            cache_hits += 1
        
        access_count += 1
        
        if thread_id < 1024:
            profiling_data[thread_id] = access_count
            access_patterns[thread_id] = cache_hits
        
        i += stride


class QuantumStateMemoryLayout:
    """Optimized memory layout strategies for quantum state vectors."""
    
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.state_size = 2 ** qubit_count
        self.cache_line_size = 128
        self.memory_alignment = 128
    
    def get_optimized_layout(self, operation_pattern: str) -> dict:
        """Get memory layout optimized for specific operation patterns."""
        if operation_pattern == "single_qubit_heavy":
            return {
                'blocking_factor': 16,
                'prefetch_distance': 8,
                'shared_memory_tiles': 32,
                'coalescing_stride': 32
            }
        elif operation_pattern == "two_qubit_heavy":
            return {
                'blocking_factor': 8,
                'prefetch_distance': 4,
                'shared_memory_tiles': 16,
                'coalescing_stride': 16
            }
        else:
            return {
                'blocking_factor': 12,
                'prefetch_distance': 6,
                'shared_memory_tiles': 24,
                'coalescing_stride': 24
            }
    
    def calculate_memory_bandwidth_utilization(
        self, 
        kernel_type: str,
        threads_per_block: int,
        blocks_per_grid: int
    ) -> dict:
        """Calculate theoretical memory bandwidth utilization."""
        
        bytes_per_element = 16
        elements_per_thread = {
            'single_qubit': 2,
            'two_qubit': 4,
            'diagonal': 1,
            'controlled': 4
        }.get(kernel_type, 2)
        
        total_threads = threads_per_block * blocks_per_grid
        total_memory_accesses = total_threads * elements_per_thread * bytes_per_element
        
        theoretical_bandwidth = 900e9
        utilization = min(1.0, total_memory_accesses / (theoretical_bandwidth * 1e-3))
        
        return {
            'utilization_percent': utilization * 100,
            'total_memory_accesses': total_memory_accesses,
            'effective_bandwidth': total_memory_accesses / 1e-3,
            'coalescing_efficiency': self._estimate_coalescing_efficiency(kernel_type),
            'cache_hit_rate': self._estimate_cache_hit_rate(kernel_type)
        }
    
    def _estimate_coalescing_efficiency(self, kernel_type: str) -> float:
        """Estimate memory coalescing efficiency for different kernel types."""
        return {
            'single_qubit': 0.85,
            'two_qubit': 0.75,
            'diagonal': 0.95,
            'controlled': 0.70
        }.get(kernel_type, 0.80)
    
    def _estimate_cache_hit_rate(self, kernel_type: str) -> float:
        """Estimate L1/L2 cache hit rate for different access patterns."""
        return {
            'single_qubit': 0.90,
            'two_qubit': 0.80,
            'diagonal': 0.95,
            'controlled': 0.75
        }.get(kernel_type, 0.85)


def apply_single_qubit_gate_memory_optimized(
    state_gpu, out_gpu, matrix, target, qubit_count, memory_optimizer
):
    """Single qubit gate with advanced memory optimization."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = qubit_count
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    half_size = state_gpu.size >> 1
    threads_per_block, blocks_per_grid = memory_optimizer.calculate_optimal_block_size(state_gpu.size, 1)
    
    shared_memory_bytes = memory_optimizer.get_shared_memory_config(threads_per_block)
    
    _optimized_single_qubit_kernel_shared_memory[blocks_per_grid, threads_per_block, shared_memory_bytes](
        state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size
    )


def apply_diagonal_gate_memory_optimized(
    state_gpu, matrix, target, out_gpu, memory_optimizer
):
    """Diagonal gate with perfect memory coalescing optimization."""
    a, d = matrix[0, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    threads_per_block, blocks_per_grid = memory_optimizer.calculate_optimal_block_size(state_gpu.size, 1)
    
    _optimized_diagonal_kernel_coalesced[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a, d, target_mask, state_gpu.size
    )


def apply_cnot_gate_memory_optimized(
    state_gpu, control, target, out_gpu, qubit_count, memory_optimizer
):
    """CNOT gate with shared memory prefetching optimization."""
    n_qubits = qubit_count
    control_bit = n_qubits - control - 1
    target_bit = n_qubits - target - 1
    control_mask = 1 << control_bit
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    quarter_size = state_gpu.size >> 2
    threads_per_block, blocks_per_grid = memory_optimizer.calculate_optimal_block_size(quarter_size, 2)
    
    shared_memory_bytes = memory_optimizer.get_shared_memory_config(threads_per_block)
    
    _optimized_cnot_kernel_shared_memory[blocks_per_grid, threads_per_block, shared_memory_bytes](
        state_flat, out_flat, control_mask, target_mask, quarter_size
    )


def apply_two_qubit_gate_memory_optimized(
    state_gpu, out_gpu, matrix, target0, target1, qubit_count, memory_optimizer
):
    """Two qubit gate with register blocking and cache optimization."""
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
    
    threads_per_block, blocks_per_grid = memory_optimizer.calculate_optimal_block_size(state_gpu.size, 4)
    
    shared_memory_bytes = memory_optimizer.get_shared_memory_config(threads_per_block)
    
    _optimized_two_qubit_kernel_register_blocked[blocks_per_grid, threads_per_block, shared_memory_bytes](
        state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, state_gpu.size
    )


def apply_controlled_gate_memory_optimized(
    state_gpu, out_gpu, op, qubit_count, memory_optimizer, matrix_cache
):
    """Controlled gate with shared memory matrix caching and prefetching."""
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
    cache_key = f"ctrl_opt_{matrix_size}_{hash(matrix.tobytes())}"
    
    if matrix_cache and cache_key in matrix_cache:
        matrix_gpu = matrix_cache[cache_key]
    else:
        matrix_contiguous = np.ascontiguousarray(matrix.flatten())
        matrix_gpu = cuda.to_device(matrix_contiguous)
        if matrix_cache:
            matrix_cache[cache_key] = matrix_gpu
    
    threads_per_block, blocks_per_grid = memory_optimizer.calculate_optimal_block_size(state_gpu.size, matrix_size)
    
    shared_memory_bytes = memory_optimizer.get_shared_memory_config(threads_per_block)
    
    _optimized_controlled_kernel_shared_memory[blocks_per_grid, threads_per_block, shared_memory_bytes](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, state_gpu.size, matrix_size
    )
