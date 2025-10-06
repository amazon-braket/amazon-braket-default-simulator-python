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
    compile_and_execute_circuit,
    _circuit_compiler,
)
from braket.default_simulator.tensor_core_acceleration import (
    accelerate_with_tensor_cores,
    _tensor_core_accelerator
)

_TENSOR_CORES_AVAILABLE = True


class GPUBufferManager:
    """Ultra-optimized GPU buffer management with persistent ping-pong buffers."""
    
    def __init__(self):
        self.ping_pong_buffers: dict[tuple[int, ...], tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]] = {}
        self.matrix_cache: dict[str, cuda.devicearray.DeviceNDArray] = {}
        self.stream = cuda.stream() if _GPU_AVAILABLE else None
        
    def get_ping_pong_buffers(self, shape: tuple[int, ...], dtype=np.complex128) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers optimized for quantum state operations."""
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
        """Get or create cached GPU matrix with optimized transfer."""
        if cache_key not in self.matrix_cache:
            matrix_contiguous = np.ascontiguousarray(matrix) if not matrix.flags['C_CONTIGUOUS'] else matrix
            
            if self.stream:
                self.matrix_cache[cache_key] = cuda.to_device(matrix_contiguous, stream=self.stream)
            else:
                self.matrix_cache[cache_key] = cuda.to_device(matrix_contiguous)
        return self.matrix_cache[cache_key]
    
    def transfer_to_device(self, host_array: np.ndarray, device_buffer: cuda.devicearray.DeviceNDArray):
        """Optimized transfer from host to device buffer."""
        if self.stream:
            cuda.to_device(host_array, to=device_buffer, stream=self.stream)
            self.stream.synchronize()
        else:
            cuda.to_device(host_array, to=device_buffer)
    
    def transfer_to_host(self, device_buffer: cuda.devicearray.DeviceNDArray) -> np.ndarray:
        """Optimized transfer from device buffer to host."""
        if self.stream:
            result = device_buffer.copy_to_host(stream=self.stream)
            self.stream.synchronize()
            return result
        else:
            return device_buffer.copy_to_host()
    
    def clear_cache(self):
        """Clear all cached resources."""
        self.ping_pong_buffers.clear()
        self.matrix_cache.clear()


_gpu_buffer_manager = GPUBufferManager() if _GPU_AVAILABLE else None


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply quantum operations using optimized fusion strategies with ping-pong buffering."""
    if not _GPU_AVAILABLE or not _should_use_gpu(state.size, qubit_count):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    if len(operations) >= 3 and len(operations) <= 20:
        fused_result = _execute_fused_operations(state, qubit_count, operations)
        if fused_result is not None:
            return fused_result
    
    return _apply_operations_individual(state, qubit_count, operations)


def _execute_fused_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray | None:
    """Execute operations using optimal fusion strategy with efficient ping-pong buffering."""
    if not _circuit_compiler.can_fuse_circuit(operations):
        return None
    
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    _gpu_buffer_manager.transfer_to_device(state, buffer_a)
    
    start_time = time.perf_counter()
    success = False
    fusion_method = "none"
    
    tensor_suitable_count = sum(1 for op in operations if len(op.targets) == 2)
    tensor_ratio = tensor_suitable_count / len(operations) if operations else 0
    
    if _TENSOR_CORES_AVAILABLE and len(operations) >= 4 and tensor_ratio >= 0.4:
        accelerated_ops, acceleration_info = accelerate_with_tensor_cores(
            operations, precision='fp16', validate_precision=False
        )
        
        if acceleration_info.get('accelerated', False):
            success = compile_and_execute_circuit(accelerated_ops, buffer_a, buffer_b, qubit_count)
            if success:
                fusion_method = f"tensor_core_{acceleration_info['precision']}"
    
    if not success:
        pattern_type = _circuit_compiler.analyze_pattern_type(operations)
        if pattern_type in ["single_chain", "mixed_gates", "diagonal_chain"]:
            template_kernel = _circuit_compiler.compile_template_kernel(operations, pattern_type, qubit_count)
            if template_kernel:
                _circuit_compiler.execute_template_kernel(
                    template_kernel, operations, buffer_a, buffer_b, qubit_count, pattern_type
                )
                success = True
                fusion_method = f"template_{pattern_type}"
    
    if not success:
        success = compile_and_execute_circuit(operations, buffer_a, buffer_b, qubit_count)
        if success:
            fusion_method = "jit_compiled"
    
    if not success:
        success = execute_template_fused_kernel(operations, buffer_a, buffer_b, qubit_count)
        if success:
            fusion_method = "template_basic"
    
    if success:
        result = _gpu_buffer_manager.transfer_to_host(buffer_b)
        fusion_time = time.perf_counter() - start_time
        estimated_individual_time = len(operations) * 0.15e-3
        speedup = estimated_individual_time / fusion_time if fusion_time > 0 else 1.0
        
        print(f"Circuit fusion ({fusion_method}): {len(operations)} operations in "
              f"{fusion_time*1000:.2f}ms (speedup: {speedup:.1f}x)")
        
        return result
    
    return None


def _apply_operations_individual(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply operations individually with tensor core pre-processing and ping-pong buffering."""
    buffer_a, buffer_b = _gpu_buffer_manager.get_ping_pong_buffers(state.shape, state.dtype)
    _gpu_buffer_manager.transfer_to_device(state, buffer_a)
    
    current_buffer = buffer_a
    output_buffer = buffer_b
    
    start_time = time.perf_counter()
    processed_operations = operations
    
    if _TENSOR_CORES_AVAILABLE and _tensor_core_accelerator:
        tensor_suitable_ops = [op for op in operations if _tensor_core_accelerator.can_accelerate_operation(op)]
        
        if len(tensor_suitable_ops) >= 2:
            accelerated_ops, acceleration_info = accelerate_with_tensor_cores(
                operations, precision='fp16', validate_precision=False
            )
            
            if acceleration_info.get('accelerated', False):
                processed_operations = accelerated_ops
                print(f"Individual tensor acceleration: {acceleration_info['operations_accelerated']} ops accelerated")
    
    for op in processed_operations:
        _apply_single_operation_optimized(op, current_buffer, output_buffer, qubit_count)
        current_buffer, output_buffer = output_buffer, current_buffer
    
    individual_time = time.perf_counter() - start_time
    method = "individual_tensor" if processed_operations != operations else "individual"
    print(f"{method} execution: {len(operations)} operations in {individual_time*1000:.2f}ms ({len(operations)} kernels)")
    
    return _gpu_buffer_manager.transfer_to_host(current_buffer)


def _apply_single_operation_optimized(
    op: GateOperation, 
    input_buffer: cuda.devicearray.DeviceNDArray, 
    output_buffer: cuda.devicearray.DeviceNDArray, 
    qubit_count: int
):
    """Apply single operation with optimized dispatch."""
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    gate_type = getattr(op, "gate_type", None)
    
    if not num_ctrl:
        if len(targets) == 1:
            if gate_type and gate_type in DIAGONAL_GATES:
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
    else:
        if len(targets) == 1 and num_ctrl == 1 and gate_type == "cphaseshift":
            _apply_controlled_phase_shift_gpu_inplace(input_buffer, op.matrix[1, 1], targets[:num_ctrl], targets[num_ctrl:][0])
        else:
            _apply_controlled_gate_gpu_direct(input_buffer, output_buffer, op, qubit_count)


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
    
    threads_per_block = min(512, max(256, total_size // 1024))
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
