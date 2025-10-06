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

import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _should_use_gpu,
    _OPTIMAL_THREADS_PER_BLOCK,
    _MAX_BLOCKS_PER_GRID,
    DIAGONAL_GATES,
    _apply_single_qubit_gate_gpu_inplace,
    _apply_two_qubit_gate_gpu_inplace,
    _apply_cnot_gpu_inplace,
    _apply_swap_gpu_inplace,
    _apply_controlled_phase_shift_gpu_inplace,
    _apply_diagonal_gate_gpu_inplace,
)
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation_strategies import single_operation_strategy


class GPUBufferManager:
    """Pure ping-pong buffer management with zero-copy swaps."""
    
    def __init__(self):
        self.ping_pong_buffers: dict[tuple[int, ...], tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: dict[str, cuda.devicearray.DeviceNDArray] = {}
        
    def get_ping_pong_buffers(self, shape: tuple[int, ...], dtype=np.complex128) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers."""
        if shape not in self.ping_pong_buffers:
            buffer_a = cuda.device_array(shape, dtype=dtype)
            buffer_b = cuda.device_array(shape, dtype=dtype)
            self.ping_pong_buffers[shape] = (buffer_a, buffer_b)
        return self.ping_pong_buffers[shape]
    
    def get_cached_matrix(self, matrix: np.ndarray, cache_key: str) -> cuda.devicearray.DeviceNDArray:
        """Get or create cached GPU matrix."""
        if cache_key not in self.matrix_cache:
            self.matrix_cache[cache_key] = cuda.to_device(matrix)
        return self.matrix_cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached resources."""
        self.ping_pong_buffers.clear()
        self.matrix_cache.clear()


_gpu_buffer_manager = GPUBufferManager() if _GPU_AVAILABLE else None


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply quantum operations using pure ping-pong GPU buffering."""
    if not _GPU_AVAILABLE or not _should_use_gpu(state.size, qubit_count):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    cuda.to_device(state, to=buffer_a)
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        if not num_ctrl:
            if len(targets) == 1:
                target = targets[0]
                if gate_type and gate_type in DIAGONAL_GATES:
                    _apply_diagonal_gate_gpu_inplace(current_buffer, op.matrix, target, output_buffer)
                else:
                    _apply_single_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, target, gate_type)
            elif len(targets) == 2:
                target0, target1 = targets[0], targets[1]
                if gate_type == "cx":
                    _apply_cnot_gpu_inplace(current_buffer, target0, target1, output_buffer)
                elif gate_type == "swap":
                    _apply_swap_gpu_inplace(current_buffer, target0, target1, output_buffer)
                else:
                    _apply_two_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, target0, target1)
        else:
            if len(targets) == 1 and len(op._ctrl_modifiers) == 1 and gate_type == "cphaseshift":
                _apply_controlled_phase_shift_gpu_inplace(current_buffer, op.matrix[1, 1], targets[:num_ctrl], targets[num_ctrl:][0])
            else:
                _apply_controlled_gate_gpu_direct(current_buffer, output_buffer, op, qubit_count)
        
        current_buffer, output_buffer = output_buffer, current_buffer
    
    return current_buffer.copy_to_host()


def _apply_controlled_gate_gpu_direct(state_gpu, out_gpu, op: GateOperation, qubit_count: int):
    """GPU-only controlled gate implementation."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    matrix = op.matrix
    gate_type = getattr(op, "gate_type", None)
    
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
    cache_key = f"ctrl_{gate_type}_{matrix_size}_{hash(matrix.tobytes())}"
    matrix_gpu = _gpu_buffer_manager.get_cached_matrix(matrix.flatten(), cache_key)
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _controlled_gate_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, total_size, matrix_size
    )


@cuda.jit(inline=True, fastmath=True)
def _controlled_gate_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                           control_state_mask, n_qubits, total_size, matrix_size):
    """Ultra-optimized CUDA kernel for controlled gate application."""
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
