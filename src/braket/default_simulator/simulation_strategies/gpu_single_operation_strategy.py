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
    QuantumGateDispatcher,
    _GPU_AVAILABLE,
    _GPU_QUBIT_THRESHOLD,
    _MIN_GPU_WORK_SIZE,
    _should_use_gpu,
)
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation_strategies import single_operation_strategy


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    work_size = state.size
    if not _should_use_gpu(work_size, state.ndim):
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    gpu_arrays = [cuda.to_device(state), cuda.device_array_like(state)]
    current_idx = 0
    
    dispatcher = GPUQuantumGateDispatcher(state.ndim)
    
    try:
        for op in operations:
            targets = op.targets
            num_ctrl = len(op._ctrl_modifiers)
            
            input_ref = gpu_arrays[current_idx]
            output_ref = gpu_arrays[1 - current_idx]
            
            _, needs_swap = _multiply_matrix_gpu(
                input_ref,
                output_ref,
                op.matrix,
                targets[num_ctrl:],
                targets[:num_ctrl],
                op._ctrl_modifiers,
                dispatcher,
                gate_type=getattr(op, "gate_type", None),
            )
            
            if needs_swap:
                current_idx = 1 - current_idx
        
        return gpu_arrays[current_idx].copy_to_host()
        
    except Exception as e:
        print(f"GPU operations failed, falling back to CPU: {e}")
        return single_operation_strategy.apply_operations(state, qubit_count, operations)


class GPUQuantumGateDispatcher:
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.use_gpu = True
        
    def apply_swap(self, state_gpu, qubit_0: int, qubit_1: int, out_gpu):
        return _apply_swap_gpu_inplace(state_gpu, qubit_0, qubit_1, out_gpu)
        
    def apply_controlled_phase_shift(self, state_gpu, phase_factor: complex, controls, target: int):
        return _apply_controlled_phase_shift_gpu_inplace(state_gpu, phase_factor, controls, target)
        
    def apply_cnot(self, state_gpu, control: int, target: int, out_gpu):
        return _apply_cnot_gpu_inplace(state_gpu, control, target, out_gpu)
        
    def apply_two_qubit_gate(self, state_gpu, matrix, target0: int, target1: int, out_gpu):
        return _apply_two_qubit_gate_gpu_inplace(state_gpu, matrix, target0, target1, out_gpu)


def _multiply_matrix_gpu(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...] | None = (),
    control_state: tuple[int, ...] | None = (),
    dispatcher: GPUQuantumGateDispatcher = None,
    gate_type: str = None,
) -> tuple[None, bool]:
    
    if not controls:
        return _multiply_matrix_gpu_no_controls(
            state_gpu, out_gpu, matrix, targets, dispatcher, gate_type
        )
    
    control_state = control_state or (1,) * len(controls)
    
    out_gpu[:] = state_gpu[:]
    return _multiply_matrix_gpu_controlled(
        state_gpu, out_gpu, matrix, targets, controls, control_state, dispatcher, gate_type
    )


@cuda.jit(inline=True, fastmath=True)
def _controlled_gate_kernel(state_flat, out_flat, matrix_flat, control_mask, target_mask, 
                           control_state_mask, n_qubits, total_size, matrix_size):
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


def _multiply_matrix_gpu_controlled(
    state_gpu,
    out_gpu,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    control_state: tuple[int, ...],
    dispatcher: GPUQuantumGateDispatcher,
    gate_type: str = None,
) -> tuple[None, bool]:
    from braket.default_simulator.linalg_utils import (
        _OPTIMAL_THREADS_PER_BLOCK,
        _MAX_BLOCKS_PER_GRID,
    )
    
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
    matrix_flat = cuda.to_device(matrix.flatten())
    
    matrix_size = matrix.shape[0]
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _controlled_gate_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, matrix_flat, control_mask, target_mask,
        control_state_mask, n_qubits, total_size, matrix_size
    )
    
    return None, True


def _multiply_matrix_gpu_no_controls(
    state_gpu,
    out_gpu, 
    matrix: np.ndarray,
    targets: tuple[int, ...],
    dispatcher: GPUQuantumGateDispatcher,
    gate_type: str = None,
) -> tuple[None, bool]:
    
    if len(targets) == 1:
        return _apply_single_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix, targets[0], gate_type)
    elif len(targets) == 2:
        return _apply_two_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix, targets[0], targets[1])
    else:
        raise NotImplementedError("Multi-qubit gates beyond 2 qubits not yet implemented for GPU")


def _apply_single_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix: np.ndarray, target: int, gate_type: str = None):
    from braket.default_simulator.linalg_utils import (
        _single_qubit_gate_kernel,
        _diagonal_gate_kernel,
        DIAGONAL_GATES,
        _OPTIMAL_THREADS_PER_BLOCK,
        _MAX_BLOCKS_PER_GRID,
    )
    
    n_qubits = len(state_gpu.shape)
    
    if gate_type and gate_type in DIAGONAL_GATES:
        a, d = matrix[0, 0], matrix[1, 1]
        target_bit = n_qubits - target - 1
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
        
        return None, True
    else:
        a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        n = n_qubits - target - 1
        mask = (1 << n) - 1
        half_size = state_gpu.size >> 1
        
        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)
        
        threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
        blocks_per_grid = min((half_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
        
        _single_qubit_gate_kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, a, b, c, d, n, mask, half_size
        )
        
        return None, True


def _apply_two_qubit_gate_gpu_inplace(state_gpu, out_gpu, matrix: np.ndarray, target0: int, target1: int):
    from braket.default_simulator.linalg_utils import (
        _two_qubit_gate_kernel,
        _OPTIMAL_THREADS_PER_BLOCK,
        _MAX_BLOCKS_PER_GRID,
    )
    
    n_qubits = len(state_gpu.shape)
    total_size = 1 << n_qubits
    
    mask_0 = 1 << (n_qubits - 1 - target0)
    mask_1 = 1 << (n_qubits - 1 - target1)
    mask_both = mask_0 | mask_1
    
    m00, m01, m02, m03 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    m10, m11, m12, m13 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    m20, m21, m22, m23 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    m30, m31, m32, m33 = matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]
    
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _two_qubit_gate_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, mask_0, mask_1, mask_both, total_size
    )
    
    return None, True


def _apply_cnot_gpu_inplace(state_gpu, control: int, target: int, out_gpu):
    from braket.default_simulator.linalg_utils import (
        _cnot_kernel,
        _OPTIMAL_THREADS_PER_BLOCK,
        _MAX_BLOCKS_PER_GRID,
    )
    
    if state_gpu is not out_gpu:
        out_gpu[:] = state_gpu[:]
    
    n_qubits = len(state_gpu.shape)
    iterations = state_gpu.size >> 2
    
    target_bit_pos = n_qubits - target - 1
    control_bit_pos = n_qubits - control - 1
    
    control_stride = 1 << control_bit_pos
    target_stride = 1 << target_bit_pos
    swap_offset = target_stride
    
    state_flat = out_gpu.reshape(-1)
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((iterations + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _cnot_kernel[blocks_per_grid, threads_per_block](
        state_flat, control_stride, target_stride, swap_offset, iterations
    )
    
    return None, False


def _apply_swap_gpu_inplace(state_gpu, qubit_0: int, qubit_1: int, out_gpu):
    from braket.default_simulator.linalg_utils import (
        _swap_kernel,
        _OPTIMAL_THREADS_PER_BLOCK,
        _MAX_BLOCKS_PER_GRID,
    )
    
    if state_gpu is not out_gpu:
        out_gpu[:] = state_gpu[:]
    
    n_qubits = len(state_gpu.shape)
    iterations = state_gpu.size >> 2
    
    pos_0 = n_qubits - 1 - qubit_0
    pos_1 = n_qubits - 1 - qubit_1
    
    if pos_0 > pos_1:
        pos_0, pos_1 = pos_1, pos_0
    
    mask_0 = 1 << pos_0
    mask_1 = 1 << pos_1
    
    state_flat = out_gpu.reshape(-1)
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((iterations + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _swap_kernel[blocks_per_grid, threads_per_block](
        state_flat, pos_0, pos_1, mask_0, mask_1, iterations
    )
    
    return None, False


@cuda.jit(inline=True, fastmath=True)
def _controlled_phase_shift_kernel(state_flat, phase_factor_real, phase_factor_imag, controlled_mask, total_size):
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


def _apply_controlled_phase_shift_gpu_inplace(state_gpu, phase_factor: complex, controls, target: int):
    from braket.default_simulator.linalg_utils import (
        _OPTIMAL_THREADS_PER_BLOCK,
        _MAX_BLOCKS_PER_GRID,
    )
    
    n_qubits = len(state_gpu.shape)
    total_size = state_gpu.size
    
    controlled_mask = 0
    for c in controls:
        controlled_mask |= 1 << (n_qubits - 1 - c)
    controlled_mask |= 1 << (n_qubits - 1 - target)
    
    state_flat = state_gpu.reshape(-1)
    
    threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    blocks_per_grid = min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID)
    
    _controlled_phase_shift_kernel[blocks_per_grid, threads_per_block](
        state_flat, phase_factor.real, phase_factor.imag, controlled_mask, total_size
    )
    
    return None, False
