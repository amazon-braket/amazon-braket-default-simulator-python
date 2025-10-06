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
        
    def get_ping_pong_buffers(self, shape: tuple[int, ...], dtype=np.complex128) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get or create persistent ping-pong buffers."""
        if shape not in self.ping_pong_buffers:
            buffer_a = cuda.device_array(shape, dtype=dtype)
            buffer_b = cuda.device_array(shape, dtype=dtype)
            self.ping_pong_buffers[shape] = (buffer_a, buffer_b)
        return self.ping_pong_buffers[shape]
    
    def clear_cache(self):
        """Clear all cached resources."""
        self.ping_pong_buffers.clear()


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
                output_buffer[:] = current_buffer[:]
                _apply_controlled_gate_fallback(current_buffer, output_buffer, op, qubit_count)
        
        current_buffer, output_buffer = output_buffer, current_buffer
    
    return current_buffer.copy_to_host()


def _apply_controlled_gate_fallback(state_gpu, out_gpu, op: GateOperation, qubit_count: int):
    """Fallback for complex controlled gates."""
    from braket.default_simulator.linalg_utils import multiply_matrix, QuantumGateDispatcher
    
    cpu_state = state_gpu.copy_to_host()
    dispatcher = QuantumGateDispatcher(qubit_count, force_cpu=True)
    
    targets = op.targets
    num_ctrl = len(op._ctrl_modifiers)
    
    result = multiply_matrix(
        cpu_state,
        op.matrix,
        targets[num_ctrl:],
        targets[:num_ctrl],
        op._ctrl_modifiers,
        dispatcher=dispatcher,
    )
    
    cuda.to_device(result, to=out_gpu)
