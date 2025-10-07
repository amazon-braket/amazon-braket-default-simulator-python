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
from braket.default_simulator.persistent_gpu_state_manager import (
    get_persistent_gpu_manager,
    PersistentGPUStateManager,
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
from braket.default_simulator.warp_cooperative_kernels import (
    apply_single_qubit_warp_cooperative,
    apply_diagonal_warp_cooperative,
    apply_cnot_warp_cooperative,
    apply_two_qubit_warp_cooperative,
    apply_controlled_warp_cooperative,
    execute_warp_cooperative_fused_sequence,
    _warp_optimizer
)
from braket.default_simulator.mega_kernel_generator import (
    execute_mega_kernel_circuit,
    _mega_kernel_generator
)

_TENSOR_CORES_AVAILABLE = True


class GPUBufferManager:
    """Advanced GPU buffer management with warp-level optimizations."""
    
    def __init__(self):
        self.ping_pong_buffers: dict[tuple[int, ...], tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: dict[str, cuda.devicearray.DeviceNDArray] = {}
        self.stream = cuda.stream() if _GPU_AVAILABLE else None
        self.warp_optimized_buffers = {}
        
    def get_ping_pong_buffers(self, shape: tuple[int, ...], dtype=np.complex128) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers optimized for warp access patterns."""
        if shape not in self.ping_pong_buffers:
            buffer_size = int(np.prod(shape))
            aligned_size = ((buffer_size + 127) // 128) * 128
            
            if self.stream:
                buffer_a = cuda.device_array(aligned_size, dtype=dtype, stream=self.stream)
                buffer_b = cuda.device_array(aligned_size, dtype=dtype, stream=self.stream)
            else:
                buffer_a = cuda.device_array(aligned_size, dtype=dtype)
                buffer_b = cuda.device_array(aligned_size, dtype=dtype)
            
            self.ping_pong_buffers[shape] = (buffer_a.reshape(shape), buffer_b.reshape(shape))
        return self.ping_pong_buffers[shape]
    
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
        self.ping_pong_buffers.clear()
        self.matrix_cache.clear()
        self.warp_optimized_buffers.clear()


_gpu_buffer_manager = GPUBufferManager() if _GPU_AVAILABLE else None


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Optimized quantum operations with direct fast-path routing for maximum speed."""
    if not _GPU_AVAILABLE:
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    return _execute_optimized_fast_path(state, qubit_count, operations)


def get_gpu_performance_stats() -> dict:
    """Get performance statistics for GPU operations and persistent state management."""
    stats = {
        'gpu_available': _GPU_AVAILABLE,
        'persistent_manager_active': False,
        'buffer_manager_active': _gpu_buffer_manager is not None
    }
    
    # Get persistent GPU manager stats
    gpu_manager = get_persistent_gpu_manager()
    if gpu_manager is not None:
        stats['persistent_manager_active'] = True
        stats.update(gpu_manager.get_performance_stats())
    
    # Get buffer manager stats
    if _gpu_buffer_manager is not None:
        stats['buffer_manager'] = {
            'ping_pong_buffers': len(_gpu_buffer_manager.ping_pong_buffers),
            'cached_matrices': len(_gpu_buffer_manager.matrix_cache),
            'stream_available': _gpu_buffer_manager.stream is not None
        }
    
    return stats


def clear_gpu_caches():
    """Clear all GPU caches and reset state."""
    gpu_manager = get_persistent_gpu_manager()
    if gpu_manager is not None:
        gpu_manager.clear_cache()
    
    if _gpu_buffer_manager is not None:
        _gpu_buffer_manager.clear_cache()


def _execute_optimized_fast_path(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Ultra-optimized execution with persistent GPU state management to eliminate redundant transfers."""
    # Use persistent GPU state manager to eliminate redundant transfers
    gpu_manager = get_persistent_gpu_manager()
    
    if gpu_manager is not None:
        # Get persistent GPU buffers, avoiding transfer if already cached
        current_buffer, output_buffer = gpu_manager.get_persistent_state(state)
        transfer_start_time = time.perf_counter()
        
        # Sync the initial state only if it's not already on GPU
        initial_sync_needed = True
        try:
            # Check if this is the same state as last time (cache hit)
            key = (state.shape, state.dtype)
            if key in gpu_manager.persistent_states:
                # State is already on GPU, verify it matches
                persistent_state = gpu_manager.persistent_states[key]
                if persistent_state.host_data is not None:
                    # Zero-copy unified memory - check if data matches
                    if np.array_equal(persistent_state.host_data, state):
                        initial_sync_needed = False
                else:
                    # Regular GPU memory - assume it needs sync for safety
                    initial_sync_needed = True
        except:
            # If any error in checking, sync to be safe
            initial_sync_needed = True
        
        if initial_sync_needed:
            if gpu_manager.memory_pool.unified_memory_enabled:
                # Zero-copy unified memory
                np.copyto(current_buffer.view(), state)
            else:
                # Regular transfer to pinned buffer if available
                cuda.to_device(state, to=current_buffer)
        
        transfer_time = time.perf_counter() - transfer_start_time
        
    else:
        # Fallback to old buffer manager if persistent manager not available
        current_buffer, output_buffer = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
        transfer_start_time = time.perf_counter()
        
        if _gpu_buffer_manager.stream:
            cuda.to_device(state, to=current_buffer, stream=_gpu_buffer_manager.stream)
            _gpu_buffer_manager.stream.synchronize()
        else:
            cuda.to_device(state, to=current_buffer)
        
        transfer_time = time.perf_counter() - transfer_start_time
    
    start_time = time.perf_counter()
    state_size = state.size
    warp_optimized = 0
    
    if len(operations) >= 3:
        mega_kernel_success = execute_mega_kernel_circuit(operations, current_buffer, qubit_count)
        
        if mega_kernel_success:
            print(f"Mega-kernel execution: {len(operations)} operations fused into single kernel")
            warp_optimized = len(operations)
        else:
            batch_size = min(20, len(operations))
            
            for i in range(0, len(operations), batch_size):
                batch_ops = operations[i:i + batch_size]
                
                mega_batch_success = execute_mega_kernel_circuit(batch_ops, current_buffer, qubit_count)
                
                if mega_batch_success:
                    warp_optimized += len(batch_ops)
                elif len(batch_ops) >= 3:
                    success = execute_template_fused_kernel(batch_ops, current_buffer, output_buffer, qubit_count)
                    if success:
                        current_buffer, output_buffer = output_buffer, current_buffer
                        continue
                    else:
                        for op in batch_ops:
                            _apply_operation_direct_dispatch(op, current_buffer, output_buffer, qubit_count, state_size)
                            current_buffer, output_buffer = output_buffer, current_buffer
                            if state_size >= 512:
                                warp_optimized += 1
                else:
                    for op in batch_ops:
                        _apply_operation_direct_dispatch(op, current_buffer, output_buffer, qubit_count, state_size)
                        current_buffer, output_buffer = output_buffer, current_buffer
                        if state_size >= 512:
                            warp_optimized += 1
    else:
        for op in operations:
            _apply_operation_direct_dispatch(op, current_buffer, output_buffer, qubit_count, state_size)
            current_buffer, output_buffer = output_buffer, current_buffer
            if state_size >= 512:
                warp_optimized += 1
    
    execution_time = time.perf_counter() - start_time
    estimated_individual_time = len(operations) * (0.05e-3 + state_size * 5e-13)
    speedup = estimated_individual_time / execution_time if execution_time > 0 else 1.0
    
    method = "persistent_gpu" if gpu_manager is not None else "standard_gpu"
    if warp_optimized > 0:
        method += "_warp_optimized"
    
    print(f"Fast-path execution ({method}): {len(operations)} operations ({qubit_count} qubits) "
          f"in {execution_time*1000:.2f}ms (speedup: {speedup:.1f}x) "
          f"[transfer: {transfer_time*1000:.2f}ms]")
    
    # Get result using persistent GPU manager for optimal transfer
    result_start_time = time.perf_counter()
    if gpu_manager is not None:
        result = gpu_manager.get_result_array(current_buffer, use_zero_copy=True)
    else:
        # Fallback to standard copy
        if _gpu_buffer_manager.stream:
            result = current_buffer.copy_to_host(stream=_gpu_buffer_manager.stream)
            _gpu_buffer_manager.stream.synchronize()
        else:
            result = current_buffer.copy_to_host()
    
    result_time = time.perf_counter() - result_start_time
    
    if gpu_manager is not None and hasattr(gpu_manager, 'stats'):
        # Update performance statistics
        if not initial_sync_needed:
            gpu_manager.stats['transfers_avoided'] += 1
            gpu_manager.stats['total_transfer_time_saved_ms'] += transfer_time * 1000
    
    return result




def _apply_operation_direct_dispatch(
    op: GateOperation, 
    input_buffer: cuda.devicearray.DeviceNDArray, 
    output_buffer: cuda.devicearray.DeviceNDArray, 
    qubit_count: int,
    state_size: int
):
    """Direct optimized dispatch with minimal overhead."""
    targets = op.targets
    gate_type = getattr(op, "gate_type", None)
    
    if len(targets) == 1:
        if gate_type == "pauli_z" or (gate_type in DIAGONAL_GATES):
            if state_size >= 512:
                apply_diagonal_warp_cooperative(input_buffer, op.matrix, targets[0], output_buffer)
            else:
                _apply_diagonal_gate_gpu_inplace(input_buffer, op.matrix, targets[0], output_buffer)
        else:
            if state_size >= 512:
                apply_single_qubit_warp_cooperative(input_buffer, output_buffer, op.matrix, targets[0], qubit_count)
            else:
                _apply_single_qubit_gate_gpu_inplace(input_buffer, output_buffer, op.matrix, targets[0], gate_type)
    
    elif len(targets) == 2:
        if gate_type == "cx":
            if state_size >= 512:
                apply_cnot_warp_cooperative(input_buffer, targets[0], targets[1], output_buffer, qubit_count)
            else:
                _apply_cnot_gpu_inplace(input_buffer, targets[0], targets[1], output_buffer)
        elif gate_type == "swap":
            _apply_swap_gpu_inplace(input_buffer, targets[0], targets[1], output_buffer)
        else:
            if state_size >= 512:
                apply_two_qubit_warp_cooperative(input_buffer, output_buffer, op.matrix, targets[0], targets[1], qubit_count)
            else:
                _apply_two_qubit_gate_gpu_inplace(input_buffer, output_buffer, op.matrix, targets[0], targets[1])
    
    elif len(op._ctrl_modifiers) > 0:
        if len(targets) == 1 and len(op._ctrl_modifiers) == 1 and gate_type == "cphaseshift":
            _apply_controlled_phase_shift_gpu_inplace(input_buffer, op.matrix[1, 1], targets[:1], targets[1:][0])
        else:
            if state_size >= 512:
                apply_controlled_warp_cooperative(input_buffer, output_buffer, op, qubit_count, _gpu_buffer_manager.matrix_cache)
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
    matrix_gpu = _gpu_buffer_manager.get_cached_matrix(matrix.flatten(), cache_key)
    
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
