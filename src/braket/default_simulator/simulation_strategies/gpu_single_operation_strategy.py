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
import math

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


class MultiGPUQuantumManager:
    """Multi-GPU quantum state management for massive performance scaling."""
    
    def __init__(self):
        self.num_gpus = 0
        self.gpu_contexts = []
        self.gpu_streams = []
        self.gpu_buffer_managers = []
        
        if _GPU_AVAILABLE:
            try:
                self.num_gpus = len(cuda.gpus)
                print(f"Multi-GPU support: {self.num_gpus} GPUs detected")
                
                for gpu_id in range(self.num_gpus):
                    cuda.select_device(gpu_id)
                    context = cuda.current_context()
                    stream = cuda.stream()
                    buffer_manager = SingleGPUBufferManager(gpu_id)
                    
                    self.gpu_contexts.append(context)
                    self.gpu_streams.append(stream)
                    self.gpu_buffer_managers.append(buffer_manager)
                
                cuda.select_device(0)
            except Exception as e:
                print(f"Multi-GPU initialization failed: {e}, falling back to single GPU")
                self.num_gpus = 1
                self.gpu_buffer_managers = [SingleGPUBufferManager(0)]
        
        if self.num_gpus == 0:
            self.gpu_buffer_managers = [None]
    
    def should_use_multi_gpu(self, qubit_count: int) -> bool:
        """Determine if multi-GPU execution would be beneficial."""
        return (self.num_gpus > 1 and 
                qubit_count >= 14)
    
    def get_qubit_partition(self, qubit_count: int) -> list[tuple[int, int]]:
        """Partition qubits across available GPUs for optimal load balancing."""
        if self.num_gpus <= 1:
            return [(0, qubit_count)]
        
        qubits_per_gpu = max(1, qubit_count // self.num_gpus)
        partitions = []
        
        for gpu_id in range(self.num_gpus):
            start_qubit = gpu_id * qubits_per_gpu
            if gpu_id == self.num_gpus - 1:
                end_qubit = qubit_count
            else:
                end_qubit = min(start_qubit + qubits_per_gpu, qubit_count)
            
            if start_qubit < qubit_count:
                partitions.append((start_qubit, end_qubit))
        
        return partitions
    
    def partition_state_vector(self, state: np.ndarray, qubit_count: int) -> list[cuda.devicearray.DeviceNDArray]:
        """Distribute quantum state across multiple GPUs."""
        if self.num_gpus <= 1:
            cuda.select_device(0)
            return [cuda.to_device(state)]
        
        partitions = self.get_qubit_partition(qubit_count)
        state_flat = state.reshape(-1)
        gpu_states = []
        
        elements_per_partition = len(state_flat) // self.num_gpus
        
        for gpu_id, (start_qubit, end_qubit) in enumerate(partitions):
            cuda.select_device(gpu_id)
            
            start_idx = gpu_id * elements_per_partition
            if gpu_id == self.num_gpus - 1:
                end_idx = len(state_flat)
            else:
                end_idx = min(start_idx + elements_per_partition, len(state_flat))
            
            partition_data = state_flat[start_idx:end_idx]
            gpu_state = cuda.to_device(partition_data, stream=self.gpu_streams[gpu_id])
            gpu_states.append(gpu_state)
        
        cuda.select_device(0)
        return gpu_states
    
    def gather_state_vector(self, gpu_states: list[cuda.devicearray.DeviceNDArray], original_shape: tuple) -> np.ndarray:
        """Gather distributed state from multiple GPUs back to host."""
        if len(gpu_states) == 1:
            cuda.select_device(0)
            return gpu_states[0].copy_to_host().reshape(original_shape)
        
        result_parts = []
        for gpu_id, gpu_state in enumerate(gpu_states):
            cuda.select_device(gpu_id)
            part = gpu_state.copy_to_host()
            result_parts.append(part)
        
        cuda.select_device(0)
        full_result = np.concatenate(result_parts)
        return full_result.reshape(original_shape)


class SingleGPUBufferManager:
    """Single GPU buffer management for multi-GPU coordination."""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.ping_pong_buffers: dict[tuple[int, ...], tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: dict[str, cuda.devicearray.DeviceNDArray] = {}
        
    def get_ping_pong_buffers(self, shape: tuple[int, ...], dtype=np.complex128) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers on this GPU."""
        cuda.select_device(self.gpu_id)
        if shape not in self.ping_pong_buffers:
            buffer_a = cuda.device_array(shape, dtype=dtype)
            buffer_b = cuda.device_array(shape, dtype=dtype)
            self.ping_pong_buffers[shape] = (buffer_a, buffer_b)
        return self.ping_pong_buffers[shape]
    
    def get_cached_matrix(self, matrix: np.ndarray, cache_key: str) -> cuda.devicearray.DeviceNDArray:
        """Get or create cached GPU matrix on this GPU."""
        cuda.select_device(self.gpu_id)
        if cache_key not in self.matrix_cache:
            self.matrix_cache[cache_key] = cuda.to_device(matrix)
        return self.matrix_cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached resources on this GPU."""
        cuda.select_device(self.gpu_id)
        self.ping_pong_buffers.clear()
        self.matrix_cache.clear()


_multi_gpu_manager = MultiGPUQuantumManager()


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply quantum operations using advanced multi-GPU or single-GPU buffering."""
    if not _GPU_AVAILABLE or not _should_use_gpu(state.size, qubit_count):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    if _multi_gpu_manager.should_use_multi_gpu(qubit_count):
        return _apply_operations_multi_gpu(state, qubit_count, operations)
    else:
        return _apply_operations_single_gpu(state, qubit_count, operations)


def _apply_operations_multi_gpu(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Multi-GPU quantum circuit execution using GPU pipelining."""
    if _multi_gpu_manager.num_gpus < 2:
        return _apply_operations_single_gpu(state, qubit_count, operations)
    
    gpu_id = 0
    current_gpu_states = {}
    
    for gpu_id in range(min(2, _multi_gpu_manager.num_gpus)):
        cuda.select_device(gpu_id)
        current_gpu_states[gpu_id] = cuda.to_device(state)
    
    current_gpu = 0
    
    for i, op in enumerate(operations):
        cuda.select_device(current_gpu)
        buffer_manager = _multi_gpu_manager.gpu_buffer_managers[current_gpu]
        
        gpu_state = current_gpu_states[current_gpu]
        buffer_a, buffer_b = buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
        
        if i == 0:
            buffer_a[:] = gpu_state[:]
            current_buffer = buffer_a
            output_buffer = buffer_b
        else:
            cuda.select_device((current_gpu + 1) % 2)
            prev_gpu_state = current_gpu_states[(current_gpu + 1) % 2]
            cuda.select_device(current_gpu)
            buffer_a[:] = prev_gpu_state.copy_to_host()
            current_buffer = buffer_a
            output_buffer = buffer_b
        
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        _execute_single_operation_gpu(
            current_buffer, output_buffer, op, targets, num_ctrl, gate_type, qubit_count
        )
        
        cuda.to_device(output_buffer.copy_to_host(), to=current_gpu_states[current_gpu])
        
        current_gpu = (current_gpu + 1) % min(2, _multi_gpu_manager.num_gpus)
    
    final_gpu = (len(operations) - 1) % min(2, _multi_gpu_manager.num_gpus)
    cuda.select_device(final_gpu)
    return current_gpu_states[final_gpu].copy_to_host()


def _execute_single_operation_gpu(
    current_buffer, output_buffer, op, targets, num_ctrl, gate_type, qubit_count
):
    """Execute a single operation on current GPU."""
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


def _apply_operations_single_gpu(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Single GPU execution with optimized ping-pong buffering."""
    cuda.select_device(0)
    buffer_manager = _multi_gpu_manager.gpu_buffer_managers[0]
    buffer_a, buffer_b = buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    
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


def _can_parallelize_operations(operations: list[GateOperation], qubit_count: int) -> bool:
    """Check if operations can benefit from multi-GPU parallelization."""
    if len(operations) < 4:
        return False
    
    single_qubit_ops = sum(1 for op in operations if len(op.targets) == 1 and not op._ctrl_modifiers)
    return single_qubit_ops >= len(operations) * 0.6


def _filter_operations_for_gpu(operations: list[GateOperation], gpu_id: int, qubit_count: int) -> list[GateOperation]:
    """Filter operations relevant to specific GPU partition."""
    partitions = _multi_gpu_manager.get_qubit_partition(qubit_count)
    if gpu_id >= len(partitions):
        return []
    
    start_qubit, end_qubit = partitions[gpu_id]
    relevant_ops = []
    
    for op in operations:
        if any(start_qubit <= target < end_qubit for target in op.targets):
            relevant_ops.append(op)
    
    return relevant_ops


def _execute_operations_on_gpu(
    gpu_state: cuda.devicearray.DeviceNDArray,
    qubit_count: int,
    operations: list[GateOperation],
    buffer_manager: SingleGPUBufferManager
):
    """Execute operations on single GPU with ping-pong buffering."""
    if not operations:
        return
    
    partitions = _multi_gpu_manager.get_qubit_partition(qubit_count)
    gpu_id = buffer_manager.gpu_id
    
    if gpu_id >= len(partitions):
        return
    
    start_qubit, end_qubit = partitions[gpu_id]
    local_qubit_count = end_qubit - start_qubit
    
    buffer_a, buffer_b = buffer_manager.get_ping_pong_buffers(gpu_state.shape, gpu_state.dtype)
    buffer_a[:] = gpu_state[:]
    
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    for op in operations:
        targets = op.targets
        num_ctrl = len(op._ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)
        
        if not num_ctrl and len(targets) == 1:
            global_target = targets[0]
            if start_qubit <= global_target < end_qubit:
                local_target = global_target - start_qubit
                
                if gate_type and gate_type in DIAGONAL_GATES:
                    _apply_diagonal_gate_gpu_inplace(current_buffer, op.matrix, local_target, output_buffer)
                else:
                    _apply_single_qubit_gate_gpu_inplace(current_buffer, output_buffer, op.matrix, local_target, gate_type)
                
                current_buffer, output_buffer = output_buffer, current_buffer
    
    gpu_state[:] = current_buffer[:]


def _synchronize_multi_gpu_operations(
    operations: list[GateOperation], 
    gpu_states: list[cuda.devicearray.DeviceNDArray], 
    qubit_count: int
):
    """Synchronize multi-qubit operations across GPU boundaries."""
    cross_gpu_ops = []
    
    for op in operations:
        targets = op.targets
        if len(targets) == 2:
            partitions = _multi_gpu_manager.get_qubit_partition(qubit_count)
            
            gpu0 = _get_gpu_for_qubit(targets[0], partitions)
            gpu1 = _get_gpu_for_qubit(targets[1], partitions)
            
            if gpu0 != gpu1:
                cross_gpu_ops.append((op, gpu0, gpu1))
    
    for op, gpu0, gpu1 in cross_gpu_ops:
        _execute_cross_gpu_operation(op, gpu0, gpu1, gpu_states, qubit_count)


def _get_gpu_for_qubit(qubit: int, partitions: list[tuple[int, int]]) -> int:
    """Determine which GPU handles a specific qubit."""
    for gpu_id, (start_qubit, end_qubit) in enumerate(partitions):
        if start_qubit <= qubit < end_qubit:
            return gpu_id
    return 0


def _execute_cross_gpu_operation(
    op: GateOperation,
    gpu0: int,
    gpu1: int, 
    gpu_states: list[cuda.devicearray.DeviceNDArray],
    qubit_count: int
):
    """Execute two-qubit operation spanning multiple GPUs."""
    cuda.select_device(0)
    
    state0 = gpu_states[gpu0].copy_to_host()
    state1 = gpu_states[gpu1].copy_to_host()
    
    combined_state = np.concatenate([state0, state1])
    
    from braket.default_simulator.linalg_utils import multiply_matrix, QuantumGateDispatcher
    dispatcher = QuantumGateDispatcher(qubit_count, force_cpu=True)
    
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    
    result = multiply_matrix(
        combined_state,
        op.matrix,
        targets[num_ctrl:],
        targets[:num_ctrl],
        op._ctrl_modifiers,
        dispatcher=dispatcher,
    )
    
    split_point = len(state0)
    result0 = result[:split_point]
    result1 = result[split_point:]
    
    cuda.select_device(gpu0)
    cuda.to_device(result0, to=gpu_states[gpu0])
    
    cuda.select_device(gpu1)
    cuda.to_device(result1, to=gpu_states[gpu1])
    
    cuda.select_device(0)


def _apply_controlled_gate_gpu_direct(state_gpu, out_gpu, op: GateOperation, qubit_count: int):
    """GPU-only controlled gate implementation."""
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
    matrix_gpu = cuda.to_device(matrix.flatten())
    
    threads_per_block = 256
    blocks_per_grid = max(
        min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID),
        128
    )
    
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
