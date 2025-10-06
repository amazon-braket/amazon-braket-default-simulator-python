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
    _GPU_QUBIT_THRESHOLD,
    _MIN_GPU_WORK_SIZE,
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
    """Streamlined high-performance GPU buffer manager."""
    
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
    """Apply quantum operations using circuit-level GPU compilation."""
    if not _GPU_AVAILABLE or not _should_use_gpu(state.size, qubit_count):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    if len(operations) > 16 and qubit_count <= 22:
        return _apply_circuit_compiled(state, qubit_count, operations)
    else:
        return _apply_operations_ping_pong(state, qubit_count, operations)


def _apply_circuit_compiled(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Compile entire circuit into single GPU kernel."""
    
    state_gpu = cuda.to_device(state)
    out_gpu = cuda.device_array_like(state)
    
    if _can_fuse_circuit(operations):
        compiled_ops = _compile_operations_data(operations, qubit_count)
        _execute_fused_circuit_kernel(state_gpu, out_gpu, compiled_ops, qubit_count)
        return out_gpu.copy_to_host()
    else:
        return _apply_operations_ping_pong(state, qubit_count, operations)


def _apply_operations_ping_pong(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Standard ping-pong GPU execution."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    cuda.to_device(state, to=buffer_a)
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        _apply_gate_gpu_fast(
            current_buffer,
            output_buffer,
            op.matrix,
            targets[num_ctrl:],
            targets[:num_ctrl],
            op._ctrl_modifiers,
            qubit_count,
            gate_type,
        )
        
        current_buffer, output_buffer = output_buffer, current_buffer
    
    return current_buffer.copy_to_host()


def _can_fuse_circuit(operations: list[GateOperation]) -> bool:
    """Check if circuit can be fused into single kernel."""
    for op in operations:
        gate_type = getattr(op, "gate_type", None)
        num_ctrl = len(op._ctrl_modifiers)
        num_targets = len(op.targets) - num_ctrl
        
        if num_targets > 2 or num_ctrl > 2:
            return False
        if gate_type not in ("pauli_x", "pauli_y", "pauli_z", "h", "s", "t", "cx", "swap", "rz", "ry", "rx", None):
            return False
    return True


def _compile_operations_data(operations: list[GateOperation], qubit_count: int):
    """Compile operations into GPU-optimized data structure."""
    compiled_data = []
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        op_data = {
            'type': gate_type or 'custom',
            'targets': targets[num_ctrl:],
            'controls': targets[:num_ctrl],
            'control_state': op._ctrl_modifiers,
            'matrix': op.matrix,
        }
        compiled_data.append(op_data)
    
    return compiled_data


def _execute_fused_circuit_kernel(state_gpu, out_gpu, compiled_ops, qubit_count: int):
    """Execute entire quantum circuit in single fused GPU kernel."""
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    total_size = state_gpu.size
    
    op_count = len(compiled_ops)
    op_types = np.zeros(op_count, dtype=np.int32)
    op_targets = np.zeros((op_count, 2), dtype=np.int32)
    op_matrices = np.zeros((op_count, 4), dtype=np.complex128)
    
    for i, op_data in enumerate(compiled_ops):
        op_type = op_data['type']
        targets = op_data['targets']
        matrix = op_data['matrix']
        
        if op_type in ('pauli_x', 'pauli_y', 'pauli_z', 'h'):
            op_types[i] = {'pauli_x': 1, 'pauli_y': 2, 'pauli_z': 3, 'h': 4}[op_type]
        elif op_type == 'cx':
            op_types[i] = 5
        elif op_type == 'swap':
            op_types[i] = 6
        else:
            op_types[i] = 0
        
        if len(targets) >= 1:
            op_targets[i, 0] = targets[0]
        if len(targets) >= 2:
            op_targets[i, 1] = targets[1]
        
        if matrix.shape == (2, 2):
            op_matrices[i, :2] = [matrix[0, 0], matrix[0, 1]]
            op_matrices[i, 2:] = [matrix[1, 0], matrix[1, 1]]
    
    op_types_gpu = cuda.to_device(op_types)
    op_targets_gpu = cuda.to_device(op_targets)
    op_matrices_gpu = cuda.to_device(op_matrices)
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _fused_circuit_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, op_types_gpu, op_targets_gpu, op_matrices_gpu, 
        op_count, qubit_count, total_size
    )


def _apply_gate_gpu_fast(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    control_state: tuple[int, ...],
    qubit_count: int,
    gate_type: str,
):
    """Apply quantum gate with maximum GPU performance."""
    
    if not controls:
        if len(targets) == 1:
            target = targets[0]
            _apply_single_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix, target, gate_type)
        elif len(targets) == 2:
            target0, target1 = targets[0], targets[1]
            if gate_type == "cx":
                _apply_cnot_gpu_inplace(state_gpu, target0, target1, out_gpu)
            elif gate_type == "swap":
                _apply_swap_gpu_inplace(state_gpu, target0, target1, out_gpu)
            else:
                _apply_two_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix, target0, target1)
    else:
        if len(targets) == 1 and len(controls) == 1 and gate_type == "cphaseshift":
            _apply_controlled_phase_shift_gpu_inplace(state_gpu, matrix[1, 1], controls, targets[0])
        else:
            _apply_controlled_gate_optimized(
                state_gpu, out_gpu, matrix, targets, controls, control_state, qubit_count, gate_type
            )


def _apply_controlled_gate_optimized(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    control_state: tuple[int, ...],
    qubit_count: int,
    gate_type: str,
):
    """Optimized controlled gate implementation."""
    total_size = state_gpu.size
    
    control_mask = 0
    control_state_mask = 0
    for ctrl, state_val in zip(controls, control_state or (1,) * len(controls)):
        bit_pos = qubit_count - 1 - ctrl
        control_mask |= 1 << bit_pos
        if state_val == 1:
            control_state_mask |= 1 << bit_pos
    
    target_mask = 0
    for target in targets:
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
            temp_mask = target_mask
            bit = 0
            while temp_mask:
                if temp_mask & 1 and (i >> (n_qubits - 1 - bit)) & 1:
                    target_state |= 1 << (matrix_size - 1 - bit)
                temp_mask >>= 1
                bit += 1
            
            new_amplitude = 0j
            for j in range(matrix_size):
                matrix_element = matrix_flat[target_state * matrix_size + j]
                
                target_idx = i & ~target_mask
                temp_j = j
                bit = 0
                while temp_j:
                    if temp_j & 1:
                        target_idx |= 1 << (n_qubits - 1 - bit)
                    temp_j >>= 1
                    bit += 1
                
                new_amplitude += matrix_element * state_flat[target_idx]
            
            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _fused_circuit_kernel(state_flat, out_flat, op_types, op_targets, op_matrices, 
                         op_count, n_qubits, total_size):
    """Entire circuit executed in single GPU kernel."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        amplitude = state_flat[i]
        temp_amplitude = amplitude
        
        for op_idx in range(op_count):
            op_type = op_types[op_idx]
            target = op_targets[op_idx, 0]
            
            if op_type == 0:
                a = op_matrices[op_idx, 0]
                b = op_matrices[op_idx, 1]
                c = op_matrices[op_idx, 2]
                d = op_matrices[op_idx, 3]
                
                target_bit = n_qubits - target - 1
                target_mask = 1 << target_bit
                paired_idx = i ^ target_mask
                
                if i <= paired_idx:
                    s0 = temp_amplitude
                    s1 = state_flat[paired_idx] if i != paired_idx else temp_amplitude
                    temp_amplitude = a * s0 + b * s1
                    if i != paired_idx:
                        cuda.atomic.add(out_flat, paired_idx, c * s0 + d * s1 - state_flat[paired_idx])
            
            elif op_type == 1:
                target_bit = n_qubits - target - 1
                target_mask = 1 << target_bit
                paired_idx = i ^ target_mask
                if i <= paired_idx and i != paired_idx:
                    temp_amplitude = state_flat[paired_idx]
            
            elif op_type == 2:
                target_bit = n_qubits - target - 1
                target_mask = 1 << target_bit
                if i & target_mask:
                    temp_amplitude *= -1j
                else:
                    temp_amplitude *= 1j
            
            elif op_type == 3:
                target_bit = n_qubits - target - 1
                target_mask = 1 << target_bit
                if i & target_mask:
                    temp_amplitude *= -1
            
            elif op_type == 4:
                target_bit = n_qubits - target - 1
                target_mask = 1 << target_bit
                paired_idx = i ^ target_mask
                
                if i <= paired_idx:
                    s0 = temp_amplitude
                    s1 = state_flat[paired_idx] if i != paired_idx else temp_amplitude
                    inv_sqrt2 = 0.7071067811865476
                    temp_amplitude = inv_sqrt2 * (s0 + s1)
                    if i != paired_idx:
                        cuda.atomic.add(out_flat, paired_idx, inv_sqrt2 * (s0 - s1) - state_flat[paired_idx])
        
        out_flat[i] = temp_amplitude
        i += stride
