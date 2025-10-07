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


class PersistentGPUStateManager:
    """Persistent GPU state management with single-buffer in-place operations."""
    
    def __init__(self):
        self.persistent_states: dict[tuple[int, ...], cuda.devicearray.DeviceNDArray] = {}
        self.temporary_buffers: dict[tuple[int, ...], cuda.devicearray.DeviceNDArray] = {}
        self.matrix_cache: dict[str, cuda.devicearray.DeviceNDArray] = {}
        self.stream = cuda.stream() if _GPU_AVAILABLE else None
        
    def get_persistent_state(self, shape: tuple[int, ...], dtype=np.complex128) -> cuda.devicearray.DeviceNDArray:
        """Get or create persistent GPU state buffer for in-place operations."""
        if shape not in self.persistent_states:
            buffer_size = int(np.prod(shape))
            
            if self.stream:
                buffer = cuda.device_array(buffer_size, dtype=dtype, stream=self.stream)
            else:
                buffer = cuda.device_array(buffer_size, dtype=dtype)
            
            self.persistent_states[shape] = buffer.reshape(shape)
        return self.persistent_states[shape]
    
    def get_temporary_buffer(self, shape: tuple[int, ...], dtype=np.complex128) -> cuda.devicearray.DeviceNDArray:
        """Get temporary buffer only when in-place operation is not mathematically possible."""
        if shape not in self.temporary_buffers:
            buffer_size = int(np.prod(shape))
            
            if self.stream:
                buffer = cuda.device_array(buffer_size, dtype=dtype, stream=self.stream)
            else:
                buffer = cuda.device_array(buffer_size, dtype=dtype)
            
            self.temporary_buffers[shape] = buffer.reshape(shape)
        return self.temporary_buffers[shape]
    
    def upload_state_once(self, state: np.ndarray) -> cuda.devicearray.DeviceNDArray:
        """Upload state to GPU once at simulation start - persistent until download."""
        gpu_state = self.get_persistent_state(state.shape, state.dtype)
        
        if self.stream:
            cuda.to_device(state, to=gpu_state, stream=self.stream)
            self.stream.synchronize()
        else:
            cuda.to_device(state, to=gpu_state)
        
        return gpu_state
    
    def download_final_result(self, gpu_state: cuda.devicearray.DeviceNDArray) -> np.ndarray:
        """Download final result from GPU only at simulation end."""
        if self.stream:
            result = gpu_state.copy_to_host(stream=self.stream)
            self.stream.synchronize()
            return result
        else:
            return gpu_state.copy_to_host()
    
    def get_cached_matrix(self, matrix: np.ndarray, cache_key: str) -> cuda.devicearray.DeviceNDArray:
        """Get or create cached GPU matrix with memory alignment."""
        if cache_key not in self.matrix_cache:
            matrix_contiguous = np.ascontiguousarray(matrix) if not matrix.flags['C_CONTIGUOUS'] else matrix
            
            if self.stream:
                self.matrix_cache[cache_key] = cuda.to_device(matrix_contiguous, stream=self.stream)
            else:
                self.matrix_cache[cache_key] = cuda.to_device(matrix_contiguous)
        return self.matrix_cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached resources."""
        self.persistent_states.clear()
        self.temporary_buffers.clear()
        self.matrix_cache.clear()


_persistent_gpu_manager = PersistentGPUStateManager() if _GPU_AVAILABLE else None


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Persistent GPU operations with single-buffer in-place execution."""
    if not _GPU_AVAILABLE:
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    return _execute_persistent_gpu_path(state, qubit_count, operations)


def clear_gpu_caches():
    """Clear all GPU caches and reset state."""
    gpu_manager = get_persistent_gpu_manager()
    if gpu_manager is not None:
        gpu_manager.clear_cache()
    
    if _persistent_gpu_manager is not None:
        _persistent_gpu_manager.clear_cache()


def _execute_persistent_gpu_path(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Persistent GPU execution with single-buffer in-place operations - eliminates memory transfer bottleneck."""
    gpu_state = _persistent_gpu_manager.upload_state_once(state)
    _apply_operations_in_place(gpu_state, qubit_count, operations)
    return _persistent_gpu_manager.download_final_result(gpu_state)


def _apply_operations_in_place(
    gpu_state: cuda.devicearray.DeviceNDArray, qubit_count: int, operations: list[GateOperation]
):
    """Apply operations in-place on persistent GPU state without ping-pong buffers."""
    if len(operations) >= 3:
        mega_kernel_success = execute_mega_kernel_circuit(operations, gpu_state, qubit_count)
        if mega_kernel_success:
            return
    
    temp_buffer = None
    needs_temp_buffer = _operations_require_temporary_buffer(operations)
    
    if needs_temp_buffer:
        temp_buffer = _persistent_gpu_manager.get_temporary_buffer(gpu_state.shape, gpu_state.dtype)
    
    current_is_main = True
    
    for i, op in enumerate(operations):
        if needs_temp_buffer and not _can_apply_in_place(op):
            if current_is_main:
                _apply_operation_direct_dispatch(op, gpu_state, temp_buffer, qubit_count, gpu_state.size)
                current_is_main = False
            else:
                _apply_operation_direct_dispatch(op, temp_buffer, gpu_state, qubit_count, gpu_state.size)
                current_is_main = True
        else:
            _apply_operation_truly_inplace(op, gpu_state, qubit_count)
    
    if needs_temp_buffer and not current_is_main:
        cuda.to_device(temp_buffer.copy_to_host(), to=gpu_state)


def _operations_require_temporary_buffer(operations: list[GateOperation]) -> bool:
    """Determine if any operations require a temporary buffer (cannot be done truly in-place)."""
    for op in operations:
        if not _can_apply_in_place(op):
            return True
    return False


def _can_apply_in_place(op: GateOperation) -> bool:
    """Check if operation can be applied truly in-place without temporary buffer."""
    gate_type = getattr(op, "gate_type", None)
    
    if gate_type in DIAGONAL_GATES or gate_type == "pauli_z":
        return True
    
    if len(op.targets) == 1 and gate_type in ["rx", "ry", "rz", "h"]:
        return True
    
    return False


def _apply_operation_truly_inplace(
    op: GateOperation, gpu_state: cuda.devicearray.DeviceNDArray, qubit_count: int
):
    """Apply operation that can be done truly in-place (modifying state directly)."""
    targets = op.targets
    gate_type = getattr(op, "gate_type", None)
    
    if len(targets) == 1 and (gate_type == "pauli_z" or gate_type in DIAGONAL_GATES):
        _apply_diagonal_gate_gpu_inplace(gpu_state, op.matrix, targets[0], gpu_state)
    else:
        temp_buffer = _persistent_gpu_manager.get_temporary_buffer(gpu_state.shape, gpu_state.dtype)
        _apply_operation_direct_dispatch(op, gpu_state, temp_buffer, qubit_count, gpu_state.size)
        cuda.to_device(temp_buffer.copy_to_host(), to=gpu_state)


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
    matrix_gpu = _persistent_gpu_manager.get_cached_matrix(matrix.flatten(), cache_key)
    
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
