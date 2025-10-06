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
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

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
    """High-performance GPU buffer manager with advanced optimizations."""
    
    def __init__(self):
        self.ping_pong_buffers: Dict[Tuple[int, ...], Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: Dict[str, cuda.devicearray.DeviceNDArray] = {}
        self.streams = None
        self.stream_idx = 0
        self.batch_buffer_pool: Dict[int, List[cuda.devicearray.DeviceNDArray]] = defaultdict(list)
        self._initialized = False
        
    def get_ping_pong_buffers(self, shape: Tuple[int, ...], dtype=np.complex128) -> Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers with memory optimization."""
        if not self._initialized:
            self._initialize_gpu_resources()
            
        if shape not in self.ping_pong_buffers:
            buffer_a = cuda.device_array(shape, dtype=dtype)
            buffer_b = cuda.device_array(shape, dtype=dtype)
            self.ping_pong_buffers[shape] = (buffer_a, buffer_b)
        return self.ping_pong_buffers[shape]
    
    def get_cached_matrix(self, matrix: np.ndarray, cache_key: str = None) -> cuda.devicearray.DeviceNDArray:
        """Get or create cached GPU matrix with pinned memory optimization."""
        if cache_key is None:
            cache_key = f"matrix_{matrix.shape}_{hash(matrix.tobytes())}"
        
        if cache_key not in self.matrix_cache:
            if matrix.flags['C_CONTIGUOUS']:
                self.matrix_cache[cache_key] = cuda.to_device(matrix)
            else:
                self.matrix_cache[cache_key] = cuda.to_device(np.ascontiguousarray(matrix))
        return self.matrix_cache[cache_key]
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources lazily when first needed."""
        if not self._initialized and _GPU_AVAILABLE:
            try:
                self.streams = [cuda.stream() for _ in range(4)]
                self._initialized = True
            except Exception:
                self.streams = [None] * 4
                self._initialized = False
    
    def get_stream(self):
        """Get next available CUDA stream for parallel execution."""
        if not self._initialized:
            self._initialize_gpu_resources()
            
        if self.streams and self.streams[0] is not None:
            stream = self.streams[self.stream_idx]
            self.stream_idx = (self.stream_idx + 1) % len(self.streams)
            return stream
        return None
    
    def get_batch_buffer(self, size: int, dtype=np.complex128) -> cuda.devicearray.DeviceNDArray:
        """Get or create batch processing buffer."""
        if self.batch_buffer_pool[size]:
            return self.batch_buffer_pool[size].pop()
        else:
            return cuda.device_array(size, dtype=dtype)
    
    def return_batch_buffer(self, buffer: cuda.devicearray.DeviceNDArray):
        """Return batch buffer to pool for reuse."""
        size = buffer.size
        if len(self.batch_buffer_pool[size]) < 8:
            self.batch_buffer_pool[size].append(buffer)
    
    def clear_cache(self):
        """Clear all cached resources."""
        self.ping_pong_buffers.clear()
        self.matrix_cache.clear()
        for pool in self.batch_buffer_pool.values():
            pool.clear()
        if self.streams:
            for stream in self.streams:
                if stream is not None:
                    try:
                        stream.synchronize()
                    except Exception:
                        pass
            self.streams = None
        self._initialized = False


_gpu_buffer_manager = GPUBufferManager()


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply quantum operations using advanced GPU optimizations."""
    if not _GPU_AVAILABLE or not _should_use_gpu(state.size, qubit_count):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    if len(operations) > 3:
        return _apply_operations_batched(state, qubit_count, operations)
    else:
        return _apply_operations_sequential(state, qubit_count, operations)


def _apply_operations_sequential(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Sequential GPU execution for small operation counts."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    cuda.to_device(state, to=buffer_a)
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        
        _apply_gate_gpu_optimized(
            current_buffer,
            output_buffer,
            op.matrix,
            targets[num_ctrl:],
            targets[:num_ctrl],
            op._ctrl_modifiers,
            qubit_count,
            gate_type=getattr(op, "gate_type", None),
        )
        
        current_buffer, output_buffer = output_buffer, current_buffer
    
    return current_buffer.copy_to_host()


def _apply_operations_batched(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Batched GPU execution with operation fusion for large operation counts."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    cuda.to_device(state, to=buffer_a)
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    batches = _create_operation_batches(operations, qubit_count)
    
    for batch in batches:
        if len(batch) == 1:
            op = batch[0]
            targets = op.targets
            num_ctrl = len(op._ctrl_modifiers)
            
            _apply_gate_gpu_optimized(
                current_buffer,
                output_buffer,
                op.matrix,
                targets[num_ctrl:],
                targets[:num_ctrl],
                op._ctrl_modifiers,
                qubit_count,
                gate_type=getattr(op, "gate_type", None),
            )
        else:
            _apply_batch_operations_fused(
                current_buffer, output_buffer, batch, qubit_count
            )
        
        current_buffer, output_buffer = output_buffer, current_buffer
    
    return current_buffer.copy_to_host()


def _create_operation_batches(operations: List[GateOperation], qubit_count: int) -> List[List[GateOperation]]:
    """Create optimized batches of operations for GPU execution."""
    batches = []
    current_batch = []
    used_qubits = set()
    
    for op in operations:
        op_qubits = set(op.targets) | set(getattr(op, '_ctrl_modifiers', []))
        
        if not current_batch or not (op_qubits & used_qubits):
            current_batch.append(op)
            used_qubits.update(op_qubits)
        else:
            if len(current_batch) >= 1:
                batches.append(current_batch)
            current_batch = [op]
            used_qubits = op_qubits
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


def _apply_batch_operations_fused(
    state_gpu, out_gpu, operations: List[GateOperation], qubit_count: int
):
    """Apply multiple non-conflicting operations in a single fused kernel."""
    if len(operations) > 4:
        for op in operations:
            targets = op.targets
            num_ctrl = len(op._ctrl_modifiers)
            _apply_gate_gpu_optimized(
                state_gpu,
                out_gpu,
                op.matrix,
                targets[num_ctrl:],
                targets[:num_ctrl],
                op._ctrl_modifiers,
                qubit_count,
                gate_type=getattr(op, "gate_type", None),
            )
            state_gpu, out_gpu = out_gpu, state_gpu
        return
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    op_data = []
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        if len(targets[num_ctrl:]) == 1 and not targets[:num_ctrl]:
            target = targets[num_ctrl:][0]
            if gate_type in DIAGONAL_GATES:
                op_data.append((0, target, op.matrix[0, 0], op.matrix[1, 1], 0, 0, 0, 0))
            else:
                a, b, c, d = op.matrix[0, 0], op.matrix[0, 1], op.matrix[1, 0], op.matrix[1, 1]
                op_data.append((1, target, a, b, c, d, 0, 0))
    
    if len(op_data) == len(operations):
        total_size = state_gpu.size
        threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
        blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
        
        op_array = cuda.to_device(np.array(op_data, dtype=np.complex128))
        
        _fused_single_qubit_kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, op_array, len(op_data), qubit_count, total_size
        )
    else:
        for op in operations:
            targets = op.targets
            num_ctrl = len(op._ctrl_modifiers)
            _apply_gate_gpu_optimized(
                state_gpu,
                out_gpu,
                op.matrix,
                targets[num_ctrl:],
                targets[:num_ctrl],
                op._ctrl_modifiers,
                qubit_count,
                gate_type=getattr(op, "gate_type", None),
            )
            state_gpu, out_gpu = out_gpu, state_gpu


def _apply_gate_gpu_optimized(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...] = (),
    control_state: tuple[int, ...] = (),
    qubit_count: int = None,
    gate_type: str = None,
):
    """Apply quantum gate with maximum GPU optimization."""
    
    if not controls:
        _apply_uncontrolled_gate_gpu(state_gpu, out_gpu, matrix, targets, gate_type)
    else:
        control_state = control_state or (1,) * len(controls)
        _apply_controlled_gate_gpu(
            state_gpu, out_gpu, matrix, targets, controls, control_state, qubit_count, gate_type
        )


def _apply_uncontrolled_gate_gpu(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    gate_type: str = None,
):
    """Apply uncontrolled gate with specialized optimizations."""
    
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


def _apply_controlled_gate_gpu(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    control_state: tuple[int, ...],
    qubit_count: int,
    gate_type: str = None,
):
    """Apply controlled gate with optimized GPU implementation."""
    
    if len(targets) == 1 and len(controls) == 1 and gate_type == "cphaseshift":
        _apply_controlled_phase_shift_gpu_inplace(
            state_gpu, matrix[1, 1], controls, targets[0]
        )
        return
    
    total_size = state_gpu.size
    
    control_mask = 0
    control_state_mask = 0
    for ctrl, state_val in zip(controls, control_state):
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
                
                target_idx = i & ~target_mask
                for bit in range(matrix_size):
                    if j & (1 << bit):
                        target_idx |= (target_mask >> (matrix_size - 1 - bit))
                
                new_amplitude += matrix_element * state_flat[target_idx]
            
            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _fused_single_qubit_kernel(state_flat, out_flat, op_data, num_ops, n_qubits, total_size):
    """Fused kernel for multiple non-conflicting single-qubit operations."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        amplitude = state_flat[i]
        
        for op_idx in range(num_ops):
            op_type = int(op_data[op_idx, 0].real)
            target = int(op_data[op_idx, 1].real)
            
            target_bit = n_qubits - target - 1
            target_mask = 1 << target_bit
            
            if op_type == 0:
                a = op_data[op_idx, 2]
                d = op_data[op_idx, 3]
                if i & target_mask:
                    amplitude *= d
                else:
                    amplitude *= a
            else:
                a = op_data[op_idx, 2]
                b = op_data[op_idx, 3]  
                c = op_data[op_idx, 4]
                d = op_data[op_idx, 5]
                
                paired_idx = i ^ target_mask
                if i < paired_idx:
                    s0 = amplitude
                    s1 = state_flat[paired_idx]
                    amplitude = a * s0 + b * s1
                    out_flat[paired_idx] = c * s0 + d * s1
        
        out_flat[i] = amplitude
        i += stride


@cuda.jit(inline=True, fastmath=True)  
def _optimized_memory_copy_kernel(src, dst, size):
    """Optimized memory copy kernel with coalesced access."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < size:
        dst[i] = src[i]
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _batch_hadamard_kernel(state_flat, out_flat, targets_array, num_targets, n_qubits, total_size):
    """Specialized kernel for batch Hadamard gate application."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    inv_sqrt2 = 0.7071067811865476
    
    while i < total_size:
        amplitude = state_flat[i]
        
        for t in range(num_targets):
            target = int(targets_array[t])
            target_bit = n_qubits - target - 1
            target_mask = 1 << target_bit
            
            paired_idx = i ^ target_mask
            if i <= paired_idx:
                s0 = amplitude
                s1 = state_flat[paired_idx] if i != paired_idx else amplitude
                
                new_s0 = inv_sqrt2 * (s0 + s1)
                new_s1 = inv_sqrt2 * (s0 - s1)
                
                amplitude = new_s0
                if i != paired_idx:
                    out_flat[paired_idx] = new_s1
        
        out_flat[i] = amplitude
        i += stride
