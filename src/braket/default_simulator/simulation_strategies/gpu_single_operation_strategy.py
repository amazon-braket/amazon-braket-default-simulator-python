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
from typing import Dict, Tuple, Optional

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _GPU_QUBIT_THRESHOLD,
    _MIN_GPU_WORK_SIZE,
    _should_use_gpu,
    _OPTIMAL_THREADS_PER_BLOCK,
    _MAX_BLOCKS_PER_GRID,
    DIAGONAL_GATES,
)
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation_strategies import single_operation_strategy


class GPUBufferManager:
    """Manages persistent GPU buffers for ping-pong operations."""
    
    def __init__(self):
        self.buffers: Dict[Tuple[int, ...], Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: Dict[str, cuda.devicearray.DeviceNDArray] = {}
        
    def get_buffers(self, shape: Tuple[int, ...], dtype=np.complex128) -> Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create ping-pong buffers for given shape."""
        if shape not in self.buffers:
            buffer_a = cuda.device_array(shape, dtype=dtype)
            buffer_b = cuda.device_array(shape, dtype=dtype)
            self.buffers[shape] = (buffer_a, buffer_b)
        return self.buffers[shape]
    
    def get_cached_matrix(self, matrix: np.ndarray, gate_type: str = None) -> cuda.devicearray.DeviceNDArray:
        """Get or create cached GPU matrix."""
        cache_key = f"{gate_type}_{matrix.shape}_{hash(matrix.tobytes())}" if gate_type else f"matrix_{matrix.shape}_{hash(matrix.tobytes())}"
        
        if cache_key not in self.matrix_cache:
            self.matrix_cache[cache_key] = cuda.to_device(matrix)
        return self.matrix_cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached buffers and matrices."""
        self.buffers.clear()
        self.matrix_cache.clear()


_gpu_buffer_manager = GPUBufferManager()


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply quantum operations using optimized GPU ping-pong buffering."""
    if not _GPU_AVAILABLE or not _should_use_gpu(state.size, qubit_count):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    try:
        buffer_a, buffer_b = _gpu_buffer_manager.get_buffers(state.shape, state.dtype)
        
        buffer_a[:] = cuda.to_device(state)[:]
        current_buffer = buffer_a
        output_buffer = buffer_b
        
        dispatcher = GPUQuantumGateDispatcher(qubit_count, _gpu_buffer_manager)

        for op in operations:
            targets = op.targets
            num_ctrl = len(op._ctrl_modifiers)
            
            success = _apply_gate_gpu_optimized(
                current_buffer,
                output_buffer,
                op.matrix,
                targets[num_ctrl:],
                targets[:num_ctrl],
                op._ctrl_modifiers,
                dispatcher,
                gate_type=getattr(op, "gate_type", None),
            )
            
            if success:
                current_buffer, output_buffer = output_buffer, current_buffer
            else:
                cpu_state = current_buffer.copy_to_host()
                return single_operation_strategy.apply_operations(cpu_state, qubit_count, operations)
        
        return current_buffer.copy_to_host()
    
    except Exception as e:
        print(f"GPU execution failed ({e}), falling back to CPU")
        return single_operation_strategy.apply_operations(state, qubit_count, operations)


class GPUQuantumGateDispatcher:
    """Optimized GPU quantum gate dispatcher with ping-pong buffer support."""
    
    def __init__(self, n_qubits: int, buffer_manager: GPUBufferManager):
        self.n_qubits = n_qubits
        self.buffer_manager = buffer_manager
        
    def apply_single_qubit_gate(self, state_gpu, out_gpu, matrix: np.ndarray, target: int, gate_type: str = None) -> bool:
        """Apply single qubit gate with optimized GPU kernels."""
        if gate_type and gate_type in DIAGONAL_GATES:
            return self._apply_diagonal_gate_inplace(state_gpu, out_gpu, matrix, target)
        else:
            return self._apply_single_qubit_gate_inplace(state_gpu, out_gpu, matrix, target)
    
    def apply_two_qubit_gate(self, state_gpu, out_gpu, matrix: np.ndarray, target0: int, target1: int) -> bool:
        """Apply two qubit gate with optimized GPU kernels."""
        return self._apply_two_qubit_gate_inplace(state_gpu, out_gpu, matrix, target0, target1)
        
    def apply_swap(self, state_gpu, out_gpu, qubit_0: int, qubit_1: int) -> bool:
        """Apply SWAP gate with optimized GPU kernels."""
        return self._apply_swap_inplace(state_gpu, out_gpu, qubit_0, qubit_1)
        
    def apply_cnot(self, state_gpu, out_gpu, control: int, target: int) -> bool:
        """Apply CNOT gate with optimized GPU kernels."""
        return self._apply_cnot_inplace(state_gpu, out_gpu, control, target)
        
    def apply_controlled_phase_shift(self, state_gpu, out_gpu, phase_factor: complex, controls, target: int) -> bool:
        """Apply controlled phase shift with optimized GPU kernels."""
        return self._apply_controlled_phase_shift_inplace(state_gpu, out_gpu, phase_factor, controls, target)

    def _apply_single_qubit_gate_inplace(self, state_gpu, out_gpu, matrix: np.ndarray, target: int) -> bool:
        """Optimized single qubit gate implementation."""
        try:
            a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
            n = self.n_qubits - target - 1
            mask = (1 << n) - 1
            half_size = state_gpu.size >> 1
            
            state_flat = state_gpu.reshape(-1)
            out_flat = out_gpu.reshape(-1)
            
            threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
            blocks_per_grid = min((half_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
            
            _single_qubit_gate_kernel[blocks_per_grid, threads_per_block](
                state_flat, out_flat, a, b, c, d, n, mask, half_size
            )
            
            return True
        except Exception:
            return False

    def _apply_diagonal_gate_inplace(self, state_gpu, out_gpu, matrix: np.ndarray, target: int) -> bool:
        """Optimized diagonal gate implementation."""
        try:
            a, d = matrix[0, 0], matrix[1, 1]
            target_bit = self.n_qubits - target - 1
            target_mask = 1 << target_bit
            shifted_target_mask = target_mask - 1
            half_size = state_gpu.size >> 1
            
            state_flat = state_gpu.reshape(-1)
            out_flat = out_gpu.reshape(-1)
            
            threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
            blocks_per_grid = min((half_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
            
            _diagonal_gate_kernel[blocks_per_grid, threads_per_block](
                state_flat, out_flat, a, d, target_bit, target_mask, shifted_target_mask, half_size
            )
            
            return True
        except Exception:
            return False

    def _apply_two_qubit_gate_inplace(self, state_gpu, out_gpu, matrix: np.ndarray, target0: int, target1: int) -> bool:
        """Optimized two qubit gate implementation."""
        try:
            mask_0 = 1 << (self.n_qubits - 1 - target0)
            mask_1 = 1 << (self.n_qubits - 1 - target1)
            mask_both = mask_0 | mask_1
            total_size = 1 << self.n_qubits
            
            state_flat = state_gpu.reshape(-1)
            out_flat = out_gpu.reshape(-1)
            
            threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
            blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
            
            _two_qubit_gate_kernel[blocks_per_grid, threads_per_block](
                state_flat, out_flat, 
                matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
                matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
                matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3],
                mask_0, mask_1, mask_both, total_size
            )
            
            return True
        except Exception:
            return False

    def _apply_swap_inplace(self, state_gpu, out_gpu, qubit_0: int, qubit_1: int) -> bool:
        """Optimized SWAP gate implementation."""
        try:
            iterations = state_gpu.size >> 2
            pos_0 = self.n_qubits - 1 - qubit_0
            pos_1 = self.n_qubits - 1 - qubit_1
            
            if pos_0 > pos_1:
                pos_0, pos_1 = pos_1, pos_0
            
            out_gpu[:] = state_gpu[:]
            
            state_flat = out_gpu.reshape(-1)
            
            threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
            blocks_per_grid = min((iterations + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
            
            _swap_kernel[blocks_per_grid, threads_per_block](
                state_flat, pos_0, pos_1, 1 << pos_0, 1 << pos_1, iterations
            )
            
            return True
        except Exception:
            return False

    def _apply_cnot_inplace(self, state_gpu, out_gpu, control: int, target: int) -> bool:
        """Optimized CNOT gate implementation."""
        try:
            iterations = state_gpu.size >> 2
            target_bit_pos = self.n_qubits - target - 1
            control_bit_pos = self.n_qubits - control - 1
            
            control_stride = 1 << control_bit_pos
            swap_offset = 1 << target_bit_pos
            
            out_gpu[:] = state_gpu[:]
            
            state_flat = out_gpu.reshape(-1)
            
            threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
            blocks_per_grid = min((iterations + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
            
            _cnot_kernel[blocks_per_grid, threads_per_block](
                state_flat, control_stride, swap_offset, swap_offset, iterations
            )
            
            return True
        except Exception:
            return False

    def _apply_controlled_phase_shift_inplace(self, state_gpu, out_gpu, phase_factor: complex, controls, target: int) -> bool:
        """Optimized controlled phase shift implementation."""
        try:
            controlled_mask = (1 << (self.n_qubits - 1 - target))
            for c in controls:
                controlled_mask |= 1 << (self.n_qubits - 1 - c)
            
            phase_real = phase_factor.real
            phase_imag = phase_factor.imag
            total_size = state_gpu.size
            
            out_gpu[:] = state_gpu[:]
            
            state_flat = out_gpu.reshape(-1)
            
            threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
            blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
            
            _controlled_phase_shift_kernel[blocks_per_grid, threads_per_block](
                state_flat, phase_real, phase_imag, controlled_mask, total_size
            )
            
            return True
        except Exception:
            return False


def _apply_gate_gpu_optimized(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...] | None = (),
    control_state: tuple[int, ...] | None = (),
    dispatcher: GPUQuantumGateDispatcher = None,
    gate_type: str = None,
) -> bool:
    """Apply quantum gate with optimized GPU implementation."""
    
    if not controls:
        return _apply_gate_gpu_no_controls(
            state_gpu, out_gpu, matrix, targets, dispatcher, gate_type
        )
    
    control_state = control_state or (1,) * len(controls)
    
    return _apply_gate_gpu_controlled(
        state_gpu, out_gpu, matrix, targets, controls, control_state, dispatcher, gate_type
    )


def _apply_gate_gpu_no_controls(
    state_gpu,
    out_gpu, 
    matrix: np.ndarray,
    targets: tuple[int, ...],
    dispatcher: GPUQuantumGateDispatcher,
    gate_type: str = None,
) -> bool:
    """Apply uncontrolled quantum gate."""
    
    if len(targets) == 1:
        return dispatcher.apply_single_qubit_gate(state_gpu, out_gpu, matrix, targets[0], gate_type)
    elif len(targets) == 2:
        return dispatcher.apply_two_qubit_gate(state_gpu, out_gpu, matrix, targets[0], targets[1])
    else:
        return False


def _apply_gate_gpu_controlled(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    control_state: tuple[int, ...],
    dispatcher: GPUQuantumGateDispatcher,
    gate_type: str = None,
) -> bool:
    """Apply controlled quantum gate with optimized GPU implementation."""
    
    try:
        n_qubits = len(state_gpu.shape)
        total_size = state_gpu.size
        
        control_mask = 0
        control_state_mask = 0
        for i, (ctrl, state_val) in enumerate(zip(controls, control_state)):
            bit_pos = n_qubits - 1 - ctrl
            control_mask |= 1 << bit_pos
            if state_val == 1:
                control_state_mask |= 1 << bit_pos
        
        target_mask = 0
        for target in targets:
            target_mask |= 1 << (n_qubits - 1 - target)
        
        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)
        
        matrix_size = matrix.shape[0]
        
        threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
        blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
        
        matrix_gpu = dispatcher.buffer_manager.get_cached_matrix(matrix.flatten(), gate_type)
        
        _controlled_gate_kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, matrix_gpu, control_mask, target_mask,
            control_state_mask, n_qubits, total_size, matrix_size
        )
        
        return True
    except Exception:
        return False


@cuda.jit(inline=True, fastmath=True)
def _single_qubit_gate_kernel(state_flat, out_flat, a, b, c, d, n, mask, half_size):
    """Optimized CUDA kernel for single qubit gate application."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < half_size:
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | (1 << n)
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        out_flat[idx0] = a * s0 + b * s1
        out_flat[idx1] = c * s0 + d * s1
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _diagonal_gate_kernel(state_flat, out_flat, a, d, target_bit, target_mask, shifted_target_mask, half_size):
    """Optimized CUDA kernel for diagonal gate application."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < half_size:
        idx0 = (i & ~shifted_target_mask) << 1 | (i & shifted_target_mask)
        idx1 = idx0 | target_mask
        
        out_flat[idx0] = a * state_flat[idx0]
        out_flat[idx1] = d * state_flat[idx1]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _two_qubit_gate_kernel(state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13, 
                          m20, m21, m22, m23, m30, m31, m32, m33, 
                          mask_0, mask_1, mask_both, total_size):
    """Optimized CUDA kernel for two-qubit gate application."""
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
def _swap_kernel(state_flat, pos_0, pos_1, mask_0, mask_1, iterations):
    """Optimized CUDA kernel for SWAP gate application."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < iterations:
        base = i + ((i >> pos_0) << pos_0)
        base += (base >> pos_1) << pos_1
        
        idx0 = base | mask_1
        idx1 = base | mask_0
        
        temp = state_flat[idx0]
        state_flat[idx0] = state_flat[idx1]
        state_flat[idx1] = temp
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _cnot_kernel(state_flat, control_stride, target_stride, swap_offset, iterations):
    """Optimized CUDA kernel for CNOT gate application."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < iterations:
        idx0 = control_stride + i
        idx1 = idx0 + swap_offset
        
        temp = state_flat[idx0]
        state_flat[idx0] = state_flat[idx1]
        state_flat[idx1] = temp
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _controlled_phase_shift_kernel(state_flat, phase_factor_real, phase_factor_imag, controlled_mask, total_size):
    """Optimized CUDA kernel for controlled phase shift."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        if (i & controlled_mask) == controlled_mask:
            real_part = state_flat[i].real
            imag_part = state_flat[i].imag
            
            new_real = real_part * phase_factor_real - imag_part * phase_factor_imag
            new_imag = real_part * phase_factor_imag + imag_part * phase_factor_real
            
            state_flat[i] = complex(new_real, new_imag)
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _controlled_gate_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                           control_state_mask, n_qubits, total_size, matrix_size):
    """Optimized CUDA kernel for controlled gate application."""
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
                
                target_idx = i
                target_idx &= ~target_mask
                for bit in range(matrix_size):
                    if j & (1 << bit):
                        target_idx |= (target_mask >> (matrix_size - 1 - bit))
                
                new_amplitude += matrix_element * state_flat[target_idx]
            
            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]
        
        i += stride
