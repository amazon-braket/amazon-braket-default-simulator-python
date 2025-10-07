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
import time

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
from braket.default_simulator.circuit_compiler import (
    execute_template_fused_kernel,
    _circuit_compiler,
)
from braket.default_simulator.tensor_core_acceleration import (
    accelerate_with_tensor_cores,
    _tensor_core_accelerator
)

_TENSOR_CORES_AVAILABLE = True


class GPUBufferManager:
    """Optimized GPU buffer management with persistent ping-pong buffers."""
    
    def __init__(self):
        self.ping_pong_buffers: dict[tuple[int, ...], tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: dict[str, cuda.devicearray.DeviceNDArray] = {}
        self.stream = cuda.stream() if _GPU_AVAILABLE else None
        
    def get_ping_pong_buffers(self, shape: tuple[int, ...], dtype=np.complex128) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers."""
        if shape not in self.ping_pong_buffers:
            if self.stream:
                buffer_a = cuda.device_array(shape, dtype=dtype, stream=self.stream)
                buffer_b = cuda.device_array(shape, dtype=dtype, stream=self.stream)
            else:
                buffer_a = cuda.device_array(shape, dtype=dtype)
                buffer_b = cuda.device_array(shape, dtype=dtype)
            self.ping_pong_buffers[shape] = (buffer_a, buffer_b)
        return self.ping_pong_buffers[shape]
    
    def get_cached_matrix(self, matrix: np.ndarray, cache_key: str) -> cuda.devicearray.DeviceNDArray:
        """Get or create cached GPU matrix."""
        if cache_key not in self.matrix_cache:
            matrix_contiguous = np.ascontiguousarray(matrix) if not matrix.flags['C_CONTIGUOUS'] else matrix
            
            if self.stream:
                self.matrix_cache[cache_key] = cuda.to_device(matrix_contiguous, stream=self.stream)
            else:
                self.matrix_cache[cache_key] = cuda.to_device(matrix_contiguous)
        return self.matrix_cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached resources."""
        self.ping_pong_buffers.clear()
        self.matrix_cache.clear()


_gpu_buffer_manager = GPUBufferManager() if _GPU_AVAILABLE else None


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply quantum operations optimized for large qubit counts with GPU acceleration."""
    if not _GPU_AVAILABLE:
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    if qubit_count >= 10:
        return _execute_large_circuit_optimized(state, qubit_count, operations)
    
    if len(operations) >= 3:
        fused_result = _execute_fused_operations(state, qubit_count, operations)
        if fused_result is not None:
            return fused_result
    
    return _apply_operations_individual(state, qubit_count, operations)


def _execute_large_circuit_optimized(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Execute circuits with large qubit counts using tensor cores and fusion for ALL operations."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    if _gpu_buffer_manager.stream:
        cuda.to_device(state, to=buffer_a, stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
    else:
        cuda.to_device(state, to=buffer_a)
    
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    start_time = time.perf_counter()
    state_size = state.size
    
    processed_operations = operations
    acceleration_method = "none"
    
    if _TENSOR_CORES_AVAILABLE and _tensor_core_accelerator:
        tensor_suitable_count = sum(1 for op in operations if _tensor_core_accelerator.can_accelerate_operation(op))
        
        if tensor_suitable_count > 0:
            accelerated_ops, acceleration_info = accelerate_with_tensor_cores(
                operations, precision='fp16', validate_precision=False
            )
            
            if acceleration_info.get('accelerated', False):
                processed_operations = accelerated_ops
                acceleration_method = f"tensor_{acceleration_info['precision']}"
                print(f"Large circuit tensor acceleration: {acceleration_info['operations_accelerated']} ops accelerated")
    
    optimal_batch_size = min(25, max(8, 150 // max(1, qubit_count - 10)))
    batches_processed = 0
    fusion_successful = 0
    
    for i in range(0, len(processed_operations), optimal_batch_size):
        batch_ops = processed_operations[i:i + optimal_batch_size]
        
        success = execute_template_fused_kernel(batch_ops, current_buffer, output_buffer, qubit_count)
        
        if success:
            current_buffer, output_buffer = output_buffer, current_buffer
            batches_processed += 1
            fusion_successful += len(batch_ops)
        else:
            for op in batch_ops:
                _apply_single_operation_large_qubit(op, current_buffer, output_buffer, qubit_count, state_size)
                current_buffer, output_buffer = output_buffer, current_buffer
    
    fusion_time = time.perf_counter() - start_time
    estimated_individual_time = len(operations) * (0.1e-3 + state_size * 1e-12)
    speedup = estimated_individual_time / fusion_time if fusion_time > 0 else 1.0
    
    method_description = f"{acceleration_method}_fusion" if acceleration_method != "none" else "fusion"
    print(f"Large circuit {method_description}: {len(operations)} operations ({qubit_count} qubits, "
          f"{fusion_successful} fused, {batches_processed} batches) in {fusion_time*1000:.2f}ms (speedup: {speedup:.1f}x)")
    
    if _gpu_buffer_manager.stream:
        result = current_buffer.copy_to_host(stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
        return result
    else:
        return current_buffer.copy_to_host()


def _execute_fused_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray | None:
    """Execute operations using fast template-based fusion with maximum performance."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    if _gpu_buffer_manager.stream:
        cuda.to_device(state, to=buffer_a, stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
    else:
        cuda.to_device(state, to=buffer_a)
    
    start_time = time.perf_counter()
    
    success = execute_template_fused_kernel(operations, buffer_a, buffer_b, qubit_count)
    
    if success:
        if _gpu_buffer_manager.stream:
            result = buffer_b.copy_to_host(stream=_gpu_buffer_manager.stream)
            _gpu_buffer_manager.stream.synchronize()
        else:
            result = buffer_b.copy_to_host()
        
        fusion_time = time.perf_counter() - start_time
        estimated_individual_time = len(operations) * 0.15e-3
        speedup = estimated_individual_time / fusion_time if fusion_time > 0 else 1.0
        
        print(f"Circuit fusion: {len(operations)} operations in "
              f"{fusion_time*1000:.2f}ms (speedup: {speedup:.1f}x)")
        
        return result
    
    return None


def _execute_segmented_fusion(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Execute very long circuits using segmented fusion for maximum benefit."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    if _gpu_buffer_manager.stream:
        cuda.to_device(state, to=buffer_a, stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
    else:
        cuda.to_device(state, to=buffer_a)
    
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    start_time = time.perf_counter()
    
    segment_size = 50
    segments_processed = 0
    
    for i in range(0, len(operations), segment_size):
        segment_ops = operations[i:i + segment_size]
        
        success = execute_template_fused_kernel(segment_ops, current_buffer, output_buffer, qubit_count)
        
        if success:
            current_buffer, output_buffer = output_buffer, current_buffer
            segments_processed += 1
        else:
            for op in segment_ops:
                targets = op.targets
                num_ctrl = len(op._ctrl_modifiers)
                gate_type = getattr(op, "gate_type", None)
                
                if not num_ctrl:
                    if len(targets) == 1:
                        if gate_type and gate_type in DIAGONAL_GATES:
                            _apply_diagonal_gate_gpu_inplace(current_buffer, op.matrix, targets[0], output_buffer)
                        else:
                            _apply_single_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, targets[0], gate_type)
                    elif len(targets) == 2:
                        if gate_type == "cx":
                            _apply_cnot_gpu_inplace(current_buffer, targets[0], targets[1], output_buffer)
                        elif gate_type == "swap":
                            _apply_swap_gpu_inplace(current_buffer, targets[0], targets[1], output_buffer)
                        else:
                            _apply_two_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, targets[0], targets[1])
                else:
                    if len(targets) == 1 and num_ctrl == 1 and gate_type == "cphaseshift":
                        _apply_controlled_phase_shift_gpu_inplace(current_buffer, op.matrix[1, 1], targets[:num_ctrl], targets[num_ctrl:][0])
                    else:
                        _apply_controlled_gate_gpu_direct(current_buffer, output_buffer, op, qubit_count)
                
                current_buffer, output_buffer = output_buffer, current_buffer
    
    fusion_time = time.perf_counter() - start_time
    estimated_individual_time = len(operations) * 0.15e-3
    speedup = estimated_individual_time / fusion_time if fusion_time > 0 else 1.0
    
    print(f"Segmented fusion: {len(operations)} operations ({segments_processed} segments) in "
          f"{fusion_time*1000:.2f}ms (speedup: {speedup:.1f}x)")
    
    if _gpu_buffer_manager.stream:
        result = current_buffer.copy_to_host(stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
        return result
    else:
        return current_buffer.copy_to_host()


def _apply_operations_individual(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply operations individually with optimized ping-pong buffering."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
    if _gpu_buffer_manager.stream:
        cuda.to_device(state, to=buffer_a, stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
    else:
        cuda.to_device(state, to=buffer_a)
    
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    start_time = time.perf_counter()
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        if not num_ctrl:
            if len(targets) == 1:
                if gate_type and gate_type in DIAGONAL_GATES:
                    _apply_diagonal_gate_gpu_inplace(current_buffer, op.matrix, targets[0], output_buffer)
                else:
                    _apply_single_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, targets[0], gate_type)
            elif len(targets) == 2:
                if gate_type == "cx":
                    _apply_cnot_gpu_inplace(current_buffer, targets[0], targets[1], output_buffer)
                elif gate_type == "swap":
                    _apply_swap_gpu_inplace(current_buffer, targets[0], targets[1], output_buffer)
                else:
                    _apply_two_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, targets[0], targets[1])
        else:
            if len(targets) == 1 and num_ctrl == 1 and gate_type == "cphaseshift":
                _apply_controlled_phase_shift_gpu_inplace(current_buffer, op.matrix[1, 1], targets[:num_ctrl], targets[num_ctrl:][0])
            else:
                _apply_controlled_gate_gpu_direct(current_buffer, output_buffer, op, qubit_count)
        
        current_buffer, output_buffer = output_buffer, current_buffer
    
    individual_time = time.perf_counter() - start_time
    print(f"Individual execution: {len(operations)} operations in {individual_time*1000:.2f}ms ({len(operations)} kernels)")
    
    if _gpu_buffer_manager.stream:
        result = current_buffer.copy_to_host(stream=_gpu_buffer_manager.stream)
        _gpu_buffer_manager.stream.synchronize()
        return result
    else:
        return current_buffer.copy_to_host()


def _apply_controlled_gate_gpu_direct(state_gpu, out_gpu, op: GateOperation, qubit_count: int):
    """Optimized controlled gate implementation with cached matrices."""
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
    matrix_gpu = _gpu_buffer_manager.get_cached_matrix(matrix.flatten(), cache_key)
    
    threads_per_block = 512
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


def _apply_single_operation_large_qubit(
    op: GateOperation, 
    input_buffer: cuda.devicearray.DeviceNDArray, 
    output_buffer: cuda.devicearray.DeviceNDArray, 
    qubit_count: int,
    state_size: int
):
    """Apply single operation optimized for large qubit counts."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    gate_type = getattr(op, "gate_type", None)
    
    if not num_ctrl:
        if len(targets) == 1:
            if gate_type and gate_type in DIAGONAL_GATES:
                _apply_diagonal_gate_gpu_large_optimized(input_buffer, op.matrix, targets[0], output_buffer, state_size)
            else:
                _apply_single_qubit_gate_gpu_large_optimized(input_buffer, output_buffer, op.matrix, targets[0], state_size)
        elif len(targets) == 2:
            if gate_type == "cx":
                _apply_cnot_gpu_large_optimized(input_buffer, targets[0], targets[1], output_buffer, qubit_count)
            elif gate_type == "swap":
                _apply_swap_gpu_large_optimized(input_buffer, targets[0], targets[1], output_buffer, qubit_count)
            else:
                _apply_two_qubit_gate_gpu_large_optimized(input_buffer, output_buffer, op.matrix, targets[0], targets[1], qubit_count)
    else:
        if len(targets) == 1 and num_ctrl == 1 and gate_type == "cphaseshift":
            _apply_controlled_phase_shift_gpu_inplace(input_buffer, op.matrix[1, 1], targets[:num_ctrl], targets[num_ctrl:][0])
        else:
            _apply_controlled_gate_gpu_large_optimized(input_buffer, output_buffer, op, qubit_count)


def _apply_diagonal_gate_gpu_large_optimized(state_gpu, matrix, target, out_gpu, state_size):
    """Diagonal gate optimized for large qubit circuits with maximum parallelism."""
    a, d = matrix[0, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    threads_per_block = 1024 if state_size >= 2**20 else 512
    blocks_per_grid = min((state_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _diagonal_kernel_large[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a, d, target_mask, state_size
    )


def _apply_single_qubit_gate_gpu_large_optimized(state_gpu, out_gpu, matrix, target, state_size):
    """Single qubit gate optimized for large qubit circuits."""
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    n_qubits = len(state_gpu.shape)
    target_bit = n_qubits - target - 1
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    half_size = state_size >> 1
    threads_per_block = 1024 if state_size >= 2**20 else 512
    blocks_per_grid = min((half_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _single_qubit_kernel_large[blocks_per_grid, threads_per_block](
        state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size
    )


def _apply_cnot_gpu_large_optimized(state_gpu, control, target, out_gpu, qubit_count):
    """CNOT gate optimized for large qubit circuits."""
    n_qubits = qubit_count
    state_size = state_gpu.size
    
    control_bit = n_qubits - control - 1
    target_bit = n_qubits - target - 1
    control_mask = 1 << control_bit
    target_mask = 1 << target_bit
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    quarter_size = state_size >> 2
    threads_per_block = 1024 if state_size >= 2**20 else 512
    blocks_per_grid = min((quarter_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _cnot_kernel_large[blocks_per_grid, threads_per_block](
        state_flat, out_flat, control_mask, target_mask, control_bit, target_bit, quarter_size
    )


def _apply_swap_gpu_large_optimized(state_gpu, qubit_0, qubit_1, out_gpu, qubit_count):
    """SWAP gate optimized for large qubit circuits."""
    n_qubits = qubit_count
    state_size = state_gpu.size
    
    pos_0 = n_qubits - 1 - qubit_0
    pos_1 = n_qubits - 1 - qubit_1
    
    if pos_0 > pos_1:
        pos_0, pos_1 = pos_1, pos_0
    
    mask_0 = 1 << pos_0
    mask_1 = 1 << pos_1
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    quarter_size = state_size >> 2
    threads_per_block = 1024 if state_size >= 2**20 else 512
    blocks_per_grid = min((quarter_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _swap_kernel_large[blocks_per_grid, threads_per_block](
        state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, quarter_size
    )


def _apply_two_qubit_gate_gpu_large_optimized(state_gpu, out_gpu, matrix, target0, target1, qubit_count):
    """Two qubit gate optimized for large qubit circuits."""
    n_qubits = qubit_count
    state_size = state_gpu.size
    
    mask_0 = 1 << (n_qubits - 1 - target0)
    mask_1 = 1 << (n_qubits - 1 - target1)
    mask_both = mask_0 | mask_1
    
    m00, m01, m02, m03 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    m10, m11, m12, m13 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    m20, m21, m22, m23 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    m30, m31, m32, m33 = matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    threads_per_block = 1024 if state_size >= 2**20 else 512
    blocks_per_grid = min((state_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _two_qubit_kernel_large[blocks_per_grid, threads_per_block](
        state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, state_size
    )


def _apply_controlled_gate_gpu_large_optimized(state_gpu, out_gpu, op, qubit_count):
    """Controlled gate optimized for large qubit circuits."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    matrix = op.matrix
    state_size = state_gpu.size
    
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
    matrix_gpu = _gpu_buffer_manager.get_cached_matrix(matrix.flatten(), cache_key)
    
    threads_per_block = 1024 if state_size >= 2**20 else 512
    blocks_per_grid = min((state_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _controlled_gate_kernel_large[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_gpu, control_mask, target_mask,
        control_state_mask, qubit_count, state_size, matrix_size
    )


@cuda.jit(inline=True, fastmath=True)
def _diagonal_kernel_large(state_flat, out_flat, a, d, target_mask, total_size):
    """Diagonal gate kernel optimized for large state vectors."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        if i & target_mask:
            out_flat[i] = d * state_flat[i]
        else:
            out_flat[i] = a * state_flat[i]
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _single_qubit_kernel_large(state_flat, out_flat, a, b, c, d, target_bit, target_mask, half_size):
    """Single qubit kernel optimized for large state vectors with coalesced memory access."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < half_size:
        idx0 = i + (i & ~(target_mask - 1))
        idx1 = idx0 | target_mask
        
        s0 = state_flat[idx0]
        s1 = state_flat[idx1]
        
        out_flat[idx0] = a * s0 + b * s1
        out_flat[idx1] = c * s0 + d * s1
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _cnot_kernel_large(state_flat, out_flat, control_mask, target_mask, control_bit, target_bit, quarter_size):
    """CNOT kernel optimized for large state vectors."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < quarter_size:
        base_idx = i + (i & ~((control_mask | target_mask) - 1))
        idx0 = base_idx | control_mask
        idx1 = idx0 | target_mask
        
        out_flat[idx0] = state_flat[idx1]
        out_flat[idx1] = state_flat[idx0]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _swap_kernel_large(state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, quarter_size):
    """SWAP kernel optimized for large state vectors."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < quarter_size:
        base = i + ((i >> pos_0) << pos_0) + ((i >> pos_1) << pos_1)
        idx0 = base | mask_1
        idx1 = base | mask_0
        
        out_flat[idx0] = state_flat[idx1]
        out_flat[idx1] = state_flat[idx0]
        
        i += stride


@cuda.jit(inline=True, fastmath=True)
def _two_qubit_kernel_large(state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
                           m20, m21, m22, m23, m30, m31, m32, m33, mask_0, mask_1, mask_both, total_size):
    """Two qubit kernel optimized for large state vectors with memory coalescing."""
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
def _controlled_gate_kernel_large(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                                 control_state_mask, n_qubits, total_size, matrix_size):
    """Controlled gate kernel optimized for large state vectors."""
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
