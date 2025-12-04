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
Optimized GPU operations with minimized host-device memory transfers.

Key optimizations:
1. Single host→GPU transfer at circuit start, single transfer back at end
2. Matrix caching with LRU eviction to avoid repeated uploads
3. Pinned memory for faster transfers on large states
4. Adaptive thread/block configuration based on problem size
5. Operation fusion for consecutive single-qubit gates
6. Asynchronous execution with CUDA streams
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
    DIAGONAL_GATES,
)

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation

_PINNED_MEMORY_THRESHOLD = 2**18
_FUSION_ENABLED = True


class _FusedOperation:
    """Represents multiple fused single-qubit gates."""
    __slots__ = ('target', 'matrix')
    
    def __init__(self, target: int, matrix: np.ndarray):
        self.target = target
        self.matrix = matrix


def _get_optimal_config(total_size: int) -> tuple[int, int]:
    """Get optimal thread/block configuration based on problem size."""
    if total_size >= 2**24:
        threads = 256
    elif total_size >= 2**20:
        threads = 512
    else:
        threads = 256
    
    blocks = min((total_size + threads - 1) // threads, _MAX_BLOCKS_PER_GRID)
    return blocks, threads


class GPUMatrixCache:
    """Thread-safe cache for GPU-resident gate matrices to avoid repeated transfers."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = {}
                    cls._instance._access_count = {}
                    cls._instance._max_entries = 256
        return cls._instance

    def get_or_upload(
        self, matrix: np.ndarray, cache_key: str = None
    ) -> cuda.devicearray.DeviceNDArray:
        """Get matrix from cache or upload to GPU."""
        if cache_key is None:
            cache_key = hash(matrix.tobytes())

        if cache_key in self._cache:
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._cache[cache_key]

        if len(self._cache) >= self._max_entries:
            self._evict_least_used()

        matrix_flat = np.ascontiguousarray(matrix.flatten())
        gpu_matrix = cuda.to_device(matrix_flat)
        self._cache[cache_key] = gpu_matrix
        self._access_count[cache_key] = 1
        return gpu_matrix

    def _evict_least_used(self):
        """Evict least frequently used entries."""
        if not self._access_count:
            return
        min_key = min(self._access_count, key=self._access_count.get)
        del self._cache[min_key]
        del self._access_count[min_key]

    def clear(self):
        """Clear all cached matrices."""
        self._cache.clear()
        self._access_count.clear()


class OptimizedGPUExecutor:
    """
    Optimized GPU executor that minimizes host-device transfers.

    Key optimizations:
    1. Single transfer to GPU at start, single transfer back at end
    2. Matrix caching to avoid repeated uploads
    3. Pinned memory for faster transfers on large states
    4. Operation fusion for consecutive single-qubit gates on same target
    5. Asynchronous kernel execution with CUDA streams
    """

    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.matrix_cache = GPUMatrixCache()
        self._stream = None
        self._pinned_buffer = None

    def execute_circuit(
        self, state: np.ndarray, operations: list[GateOperation]
    ) -> np.ndarray:
        """
        Execute all operations on GPU with minimal memory transfers.

        Args:
            state: Initial state vector (on host)
            operations: List of gate operations to apply

        Returns:
            Final state vector (on host)
        """
        if not operations:
            return state

        self._stream = cuda.stream()
        
        use_pinned = state.size >= _PINNED_MEMORY_THRESHOLD
        
        if use_pinned:
            state_contiguous = np.ascontiguousarray(state)
            with cuda.pinned(state_contiguous):
                gpu_state = cuda.to_device(state_contiguous, stream=self._stream)
        else:
            state_contiguous = np.ascontiguousarray(state)
            gpu_state = cuda.to_device(state_contiguous, stream=self._stream)
        
        gpu_temp = cuda.device_array_like(gpu_state, stream=self._stream)
        self._stream.synchronize()

        current_buffer = gpu_state
        temp_buffer = gpu_temp

        if _FUSION_ENABLED:
            operations = self._fuse_operations(operations)

        for op in operations:
            if isinstance(op, _FusedOperation):
                needs_swap = self._apply_fused(op, current_buffer, temp_buffer)
            else:
                needs_swap = self._apply_operation(op, current_buffer, temp_buffer)
            if needs_swap:
                current_buffer, temp_buffer = temp_buffer, current_buffer

        self._stream.synchronize()
        
        if use_pinned:
            result = np.empty_like(state_contiguous)
            with cuda.pinned(result):
                current_buffer.copy_to_host(result, stream=self._stream)
                self._stream.synchronize()
        else:
            result = current_buffer.copy_to_host()

        return result
    
    def _fuse_operations(self, operations: list) -> list:
        """Fuse consecutive single-qubit gates on the same target."""
        if len(operations) < 2:
            return operations
        
        fused = []
        pending: dict[int, list] = {}
        
        for op in operations:
            targets = op.targets
            ctrl_modifiers = getattr(op, "_ctrl_modifiers", [])
            
            if len(targets) == 1 and len(ctrl_modifiers) == 0:
                target = targets[0]
                if target in pending:
                    pending[target].append(op.matrix)
                    if len(pending[target]) >= 4:
                        fused_matrix = pending[target][0]
                        for m in pending[target][1:]:
                            fused_matrix = m @ fused_matrix
                        fused.append(_FusedOperation(target, fused_matrix))
                        del pending[target]
                else:
                    pending[target] = [op.matrix]
            else:
                for t, matrices in pending.items():
                    if len(matrices) > 1:
                        fused_matrix = matrices[0]
                        for m in matrices[1:]:
                            fused_matrix = m @ fused_matrix
                        fused.append(_FusedOperation(t, fused_matrix))
                    else:
                        fused.append(_FusedOperation(t, matrices[0]))
                pending.clear()
                fused.append(op)
        
        for t, matrices in pending.items():
            if len(matrices) > 1:
                fused_matrix = matrices[0]
                for m in matrices[1:]:
                    fused_matrix = m @ fused_matrix
                fused.append(_FusedOperation(t, fused_matrix))
            else:
                fused.append(_FusedOperation(t, matrices[0]))
        
        return fused
    
    def _apply_fused(
        self,
        op: _FusedOperation,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
    ) -> bool:
        """Apply fused single-qubit operation."""
        return self._apply_single_qubit(state_gpu, out_gpu, op.matrix, op.target, None)

    def _apply_operation(
        self,
        op: GateOperation,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
    ) -> bool:
        """Apply single operation on GPU, return True if buffers should swap."""
        targets = op.targets
        ctrl_modifiers = getattr(op, "_ctrl_modifiers", [])
        num_ctrl = len(ctrl_modifiers)
        gate_type = getattr(op, "gate_type", None)

        actual_targets = targets[num_ctrl:]
        controls = targets[:num_ctrl]

        if len(actual_targets) == 1 and not controls:
            return self._apply_single_qubit(
                state_gpu, out_gpu, op.matrix, actual_targets[0], gate_type
            )

        elif len(actual_targets) == 2 and not controls:
            if gate_type == "cx" or gate_type == "cnot":
                return self._apply_cnot(
                    state_gpu, out_gpu, actual_targets[0], actual_targets[1]
                )
            elif gate_type == "cz":
                return self._apply_cz(
                    state_gpu, out_gpu, actual_targets[0], actual_targets[1]
                )
            elif gate_type == "swap":
                return self._apply_swap(
                    state_gpu, out_gpu, actual_targets[0], actual_targets[1]
                )
            else:
                return self._apply_two_qubit(
                    state_gpu, out_gpu, op.matrix, actual_targets[0], actual_targets[1]
                )

        elif controls:
            return self._apply_controlled(
                state_gpu, out_gpu, op, controls, actual_targets, ctrl_modifiers
            )

        return False

    def _apply_single_qubit(
        self,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        matrix: np.ndarray,
        target: int,
        gate_type: str = None,
    ) -> bool:
        """Apply single qubit gate with optimized kernel matching _large patterns."""
        n_qubits = len(state_gpu.shape)
        target_bit = n_qubits - target - 1
        half_size = state_gpu.size >> 1
        mask = (1 << target_bit) - 1

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        blocks, threads = _get_optimal_config(half_size)

        if gate_type == "pauli_x" or gate_type == "x":
            _x_gate_kernel[blocks, threads, self._stream](
                state_flat, out_flat, target_bit, mask, half_size
            )
        elif gate_type == "pauli_y" or gate_type == "y":
            _y_gate_kernel[blocks, threads, self._stream](
                state_flat, out_flat, target_bit, mask, half_size
            )
        elif gate_type == "pauli_z" or gate_type == "z":
            _z_gate_kernel[blocks, threads, self._stream](
                state_flat, out_flat, target_bit, mask, half_size
            )
        elif gate_type == "hadamard" or gate_type == "h":
            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            _h_gate_kernel[blocks, threads, self._stream](
                state_flat, out_flat, target_bit, mask, half_size, inv_sqrt2
            )
        elif gate_type and gate_type in DIAGONAL_GATES:
            a, d = matrix[0, 0], matrix[1, 1]
            _diagonal_kernel[blocks, threads, self._stream](
                state_flat, out_flat, a, d, target_bit, mask, half_size
            )
        else:
            a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
            _single_qubit_kernel[blocks, threads, self._stream](
                state_flat, out_flat, a, b, c, d, target_bit, mask, half_size
            )

        return True

    def _apply_cnot(
        self,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        control: int,
        target: int,
    ) -> bool:
        """Apply CNOT gate with optimized kernel."""
        n_qubits = len(state_gpu.shape)
        control_bit = n_qubits - control - 1
        target_bit = n_qubits - target - 1
        total_size = state_gpu.size

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        blocks, threads = _get_optimal_config(total_size)

        _cnot_kernel[blocks, threads, self._stream](
            state_flat, out_flat, 1 << control_bit, 1 << target_bit, total_size
        )

        return True

    def _apply_cz(
        self,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        control: int,
        target: int,
    ) -> bool:
        """Apply CZ gate with optimized kernel."""
        n_qubits = len(state_gpu.shape)
        control_bit = n_qubits - control - 1
        target_bit = n_qubits - target - 1
        total_size = state_gpu.size

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        blocks, threads = _get_optimal_config(total_size)

        _cz_kernel[blocks, threads, self._stream](
            state_flat, out_flat, 1 << control_bit, 1 << target_bit, total_size
        )

        return True

    def _apply_swap(
        self,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        qubit_0: int,
        qubit_1: int,
    ) -> bool:
        """Apply SWAP gate with optimized kernel."""
        n_qubits = len(state_gpu.shape)
        pos_0 = n_qubits - 1 - qubit_0
        pos_1 = n_qubits - 1 - qubit_1

        if pos_0 > pos_1:
            pos_0, pos_1 = pos_1, pos_0

        mask_0 = 1 << pos_0
        mask_1 = 1 << pos_1
        iterations = state_gpu.size >> 2

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        blocks, threads = _get_optimal_config(iterations)

        _swap_kernel[blocks, threads, self._stream](
            state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, iterations
        )

        return True

    def _apply_two_qubit(
        self,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        matrix: np.ndarray,
        target0: int,
        target1: int,
    ) -> bool:
        """Apply two-qubit gate with optimized kernel."""
        n_qubits = len(state_gpu.shape)
        pos_0 = n_qubits - 1 - target0
        pos_1 = n_qubits - 1 - target1
        if pos_0 > pos_1:
            pos_0, pos_1 = pos_1, pos_0
        mask_0 = 1 << (n_qubits - 1 - target0)
        mask_1 = 1 << (n_qubits - 1 - target1)
        mask_both = mask_0 | mask_1
        quarter_size = state_gpu.size >> 2

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        cache_key = f"2q_{hash(matrix.tobytes())}"
        matrix_gpu = self.matrix_cache.get_or_upload(matrix, cache_key)

        blocks, threads = _get_optimal_config(quarter_size)

        _two_qubit_kernel[blocks, threads, self._stream](
            state_flat, out_flat, matrix_gpu, mask_0, mask_1, mask_both, quarter_size, pos_0, pos_1
        )

        return True

    def _apply_controlled(
        self,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        op: GateOperation,
        controls: tuple,
        targets: tuple,
        ctrl_modifiers: list,
    ) -> bool:
        """Apply controlled gate with cached matrix."""
        n_qubits = len(state_gpu.shape)
        matrix = op.matrix
        total_size = state_gpu.size

        control_mask = 0
        control_state_mask = 0
        for ctrl, state_val in zip(controls, ctrl_modifiers):
            bit_pos = n_qubits - 1 - ctrl
            control_mask |= 1 << bit_pos
            if state_val == 1:
                control_state_mask |= 1 << bit_pos

        target_mask = 0
        for target in targets:
            target_mask |= 1 << (n_qubits - 1 - target)

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        cache_key = f"ctrl_{hash(matrix.tobytes())}"
        matrix_gpu = self.matrix_cache.get_or_upload(matrix, cache_key)
        matrix_size = matrix.shape[0]

        blocks, threads = _get_optimal_config(total_size)

        _controlled_kernel[blocks, threads, self._stream](
            state_flat,
            out_flat,
            matrix_gpu,
            control_mask,
            target_mask,
            control_state_mask,
            n_qubits,
            total_size,
            matrix_size,
        )

        return True


@cuda.jit(fastmath=True)
def _single_qubit_kernel(state_flat, out_flat, a, b, c, d, n, mask, half_size):
    """Single qubit gate using bit masking pattern from _apply_single_qubit_gate_large."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    target_mask = 1 << n
    for i in range(idx, half_size, stride):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | target_mask

        s0 = state_flat[idx0]
        s1 = state_flat[idx1]

        out_flat[idx0] = a * s0 + b * s1
        out_flat[idx1] = c * s0 + d * s1


@cuda.jit(fastmath=True)
def _x_gate_kernel(state_flat, out_flat, n, mask, half_size):
    """X gate using bit masking pattern from _apply_x_gate_large."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    target_mask = 1 << n
    for i in range(idx, half_size, stride):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | target_mask

        out_flat[idx0] = state_flat[idx1]
        out_flat[idx1] = state_flat[idx0]


@cuda.jit(fastmath=True)
def _h_gate_kernel(state_flat, out_flat, n, mask, half_size, inv_sqrt2):
    """Hadamard gate using bit masking pattern."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    target_mask = 1 << n
    for i in range(idx, half_size, stride):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | target_mask

        s0 = state_flat[idx0]
        s1 = state_flat[idx1]

        out_flat[idx0] = inv_sqrt2 * (s0 + s1)
        out_flat[idx1] = inv_sqrt2 * (s0 - s1)


@cuda.jit(fastmath=True)
def _diagonal_kernel(state_flat, out_flat, a, d, n, mask, half_size):
    """Diagonal gate using bit masking pattern from _apply_diagonal_gate_large."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    target_mask = 1 << n
    for i in range(idx, half_size, stride):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | target_mask

        out_flat[idx0] = a * state_flat[idx0]
        out_flat[idx1] = d * state_flat[idx1]


@cuda.jit(fastmath=True)
def _y_gate_kernel(state_flat, out_flat, n, mask, half_size):
    """Y gate: [[0, -i], [i, 0]]."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    target_mask = 1 << n
    for i in range(idx, half_size, stride):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | target_mask

        s0 = state_flat[idx0]
        s1 = state_flat[idx1]

        out_flat[idx0] = -1j * s1
        out_flat[idx1] = 1j * s0


@cuda.jit(fastmath=True)
def _z_gate_kernel(state_flat, out_flat, n, mask, half_size):
    """Z gate: [[1, 0], [0, -1]]."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    target_mask = 1 << n
    for i in range(idx, half_size, stride):
        idx0 = (i & ~mask) << 1 | (i & mask)
        idx1 = idx0 | target_mask

        out_flat[idx0] = state_flat[idx0]
        out_flat[idx1] = -state_flat[idx1]


@cuda.jit(fastmath=True)
def _cz_kernel(state_flat, out_flat, control_mask, target_mask, total_size):
    """CZ gate - applies -1 phase when both control and target are |1>."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    both_mask = control_mask | target_mask
    for i in range(idx, total_size, stride):
        if (i & both_mask) == both_mask:
            out_flat[i] = -state_flat[i]
        else:
            out_flat[i] = state_flat[i]


@cuda.jit(fastmath=True)
def _cnot_kernel(state_flat, out_flat, control_mask, target_mask, total_size):
    """CNOT gate - copies with conditional swap based on control bit."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, total_size, stride):
        if (i & control_mask) != 0:
            partner = i ^ target_mask
            out_flat[i] = state_flat[partner]
        else:
            out_flat[i] = state_flat[i]


@cuda.jit(fastmath=True)
def _swap_kernel(state_flat, out_flat, pos_0, pos_1, mask_0, mask_1, iterations):
    """SWAP gate using bit masking pattern from _apply_swap_large."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, iterations, stride):
        base = i + ((i >> pos_0) << pos_0)
        base += (base >> pos_1) << pos_1

        idx0 = base | mask_1
        idx1 = base | mask_0

        out_flat[idx0] = state_flat[idx1]
        out_flat[idx1] = state_flat[idx0]


@cuda.jit(fastmath=True)
def _two_qubit_kernel(
    state_flat, out_flat, matrix_flat, mask_0, mask_1, mask_both, quarter_size, pos_0, pos_1
):
    """Two-qubit gate - iterates only over valid base indices (1/4 of total)."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    m = cuda.shared.array(16, dtype=numba.complex128)
    if cuda.threadIdx.x < 16:
        m[cuda.threadIdx.x] = matrix_flat[cuda.threadIdx.x]
    cuda.syncthreads()

    for j in range(idx, quarter_size, stride):
        i = j
        i = (i & ~((1 << pos_0) - 1)) << 1 | (i & ((1 << pos_0) - 1))
        i = (i & ~((1 << pos_1) - 1)) << 1 | (i & ((1 << pos_1) - 1))

        s0 = state_flat[i]
        s1 = state_flat[i | mask_1]
        s2 = state_flat[i | mask_0]
        s3 = state_flat[i | mask_both]

        out_flat[i] = m[0] * s0 + m[1] * s1 + m[2] * s2 + m[3] * s3
        out_flat[i | mask_1] = m[4] * s0 + m[5] * s1 + m[6] * s2 + m[7] * s3
        out_flat[i | mask_0] = m[8] * s0 + m[9] * s1 + m[10] * s2 + m[11] * s3
        out_flat[i | mask_both] = m[12] * s0 + m[13] * s1 + m[14] * s2 + m[15] * s3


@cuda.jit(fastmath=True)
def _controlled_kernel(
    state_flat,
    out_flat,
    matrix_flat,
    control_mask,
    target_mask,
    control_state_mask,
    n_qubits,
    total_size,
    matrix_size,
):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, total_size, stride):
        if (i & control_mask) == control_state_mask:
            target_state = 0
            for bit in range(matrix_size):
                if i & (target_mask >> bit):
                    target_state |= 1 << bit

            new_amplitude = 0j
            for j in range(matrix_size):
                matrix_element = matrix_flat[target_state * matrix_size + j]

                target_idx = i & ~target_mask
                for bit in range(matrix_size):
                    if j & (1 << bit):
                        target_idx |= target_mask >> (matrix_size - 1 - bit)

                new_amplitude += matrix_element * state_flat[target_idx]

            out_flat[i] = new_amplitude
        else:
            out_flat[i] = state_flat[i]


_global_executor: OptimizedGPUExecutor | None = None
_executor_lock = threading.Lock()


def get_gpu_executor(qubit_count: int) -> OptimizedGPUExecutor:
    """Get or create GPU executor for given qubit count."""
    global _global_executor

    with _executor_lock:
        if _global_executor is None or _global_executor.qubit_count != qubit_count:
            _global_executor = OptimizedGPUExecutor(qubit_count)
        return _global_executor


def apply_operations_optimized(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """
    Apply operations with optimized GPU execution.

    This is the main entry point that replaces the existing gpu_single_operation_strategy.
    Key improvements:
    - Single host to GPU transfer at start
    - Single GPU to host transfer at end
    - Matrix caching to avoid repeated uploads
    - Efficient kernel configurations
    """
    if not _GPU_AVAILABLE or qubit_count < 8 or state.size < 256:
        from braket.default_simulator.simulation_strategies import (
            single_operation_strategy,
        )

        return single_operation_strategy.apply_operations(state, qubit_count, operations)

    executor = get_gpu_executor(qubit_count)
    return executor.execute_circuit(state, operations)


def clear_matrix_cache():
    """Clear the GPU matrix cache."""
    GPUMatrixCache().clear()


class OptimizedDensityMatrixExecutor:
    """
    GPU executor optimized for density matrix simulations.
    
    Density matrices require applying U * rho * U† which means two matrix
    multiplications per gate. This executor keeps the density matrix on GPU
    throughout the circuit to avoid per-operation transfers.
    """
    
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.matrix_cache = GPUMatrixCache()
        self._stream = None
    
    def execute_circuit(
        self, 
        density_matrix: np.ndarray, 
        operations: list
    ) -> np.ndarray:
        """
        Execute all operations on density matrix with minimal transfers.
        
        Args:
            density_matrix: Initial density matrix [2^n, 2^n] on host
            operations: List of gate operations
            
        Returns:
            Final density matrix on host
        """
        if not operations:
            return density_matrix
        
        self._stream = cuda.stream()
        
        dm_contiguous = np.ascontiguousarray(density_matrix)
        use_pinned = dm_contiguous.size >= _PINNED_MEMORY_THRESHOLD
        
        if use_pinned:
            with cuda.pinned(dm_contiguous):
                gpu_dm = cuda.to_device(dm_contiguous, stream=self._stream)
        else:
            gpu_dm = cuda.to_device(dm_contiguous, stream=self._stream)
        
        gpu_temp = cuda.device_array_like(gpu_dm, stream=self._stream)
        self._stream.synchronize()
        
        current = gpu_dm
        temp = gpu_temp
        
        for op in operations:
            needs_swap = self._apply_gate_to_dm(op, current, temp)
            if needs_swap:
                current, temp = temp, current
        
        self._stream.synchronize()
        
        if use_pinned:
            result = np.empty_like(dm_contiguous)
            with cuda.pinned(result):
                current.copy_to_host(result, stream=self._stream)
                self._stream.synchronize()
        else:
            result = current.copy_to_host()
        
        return result
    
    def _apply_gate_to_dm(
        self,
        op,
        dm_gpu: cuda.devicearray.DeviceNDArray,
        temp_gpu: cuda.devicearray.DeviceNDArray,
    ) -> bool:
        """Apply U * rho * U† to density matrix on GPU."""
        matrix = op.matrix
        targets = op.targets
        n = self.qubit_count
        dim = 1 << n
        
        dm_flat = dm_gpu.reshape(-1)
        temp_flat = temp_gpu.reshape(-1)
        
        if len(targets) == 1:
            target = targets[0]
            target_bit = n - target - 1
            total_size = dim * dim
            quarter_size = total_size >> 2
            
            blocks, threads = _get_optimal_config(quarter_size)
            
            a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
            a_conj, b_conj = np.conj(a), np.conj(b)
            c_conj, d_conj = np.conj(c), np.conj(d)
            
            _dm_single_qubit_kernel[blocks, threads, self._stream](
                dm_flat, temp_flat,
                a, b, c, d,
                a_conj, b_conj, c_conj, d_conj,
                target_bit, n, total_size
            )
            return True
        
        elif len(targets) == 2:
            target0, target1 = targets[0], targets[1]
            row_mask_0 = 1 << (n - 1 - target0)
            row_mask_1 = 1 << (n - 1 - target1)
            col_mask_0 = row_mask_0
            col_mask_1 = row_mask_1
            total_size = dim * dim
            sixteenth_size = total_size >> 4
            
            blocks, threads = _get_optimal_config(sixteenth_size)
            
            cache_key = f"dm_2q_{hash(matrix.tobytes())}"
            matrix_gpu = self.matrix_cache.get_or_upload(matrix, cache_key)
            
            matrix_conj = np.conj(matrix)
            cache_key_conj = f"dm_2q_conj_{hash(matrix_conj.tobytes())}"
            matrix_conj_gpu = self.matrix_cache.get_or_upload(matrix_conj, cache_key_conj)
            
            _dm_two_qubit_kernel[blocks, threads, self._stream](
                dm_flat, temp_flat,
                matrix_gpu, matrix_conj_gpu,
                row_mask_0, row_mask_1, col_mask_0, col_mask_1,
                n, total_size
            )
            return True
        
        return False


@cuda.jit(fastmath=True)
def _dm_single_qubit_kernel(
    dm_flat, out_flat,
    a, b, c, d,
    a_conj, b_conj, c_conj, d_conj,
    target_bit, dim_log2, total_size
):
    """
    Apply U * rho * U† for single-qubit gate on density matrix.
    Uses bit masking pattern from _apply_single_qubit_gate_large.
    
    Iterates over all elements where both row and col target bits are 0,
    then processes the 2x2 block.
    """
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    dim = 1 << dim_log2
    row_target = 1 << target_bit
    col_target = row_target
    quarter_size = total_size >> 2
    
    for i in range(idx, quarter_size, stride):
        flat_i = i
        flat_i = (flat_i & ~((1 << target_bit) - 1)) << 1 | (flat_i & ((1 << target_bit) - 1))
        flat_i = (flat_i & ~((1 << (target_bit + dim_log2)) - 1)) << 1 | (flat_i & ((1 << (target_bit + dim_log2)) - 1))
        
        row_0 = flat_i >> dim_log2
        col_0 = flat_i & (dim - 1)
        row_1 = row_0 | row_target
        col_1 = col_0 | col_target
        
        idx_00 = row_0 * dim + col_0
        idx_01 = row_0 * dim + col_1
        idx_10 = row_1 * dim + col_0
        idx_11 = row_1 * dim + col_1
        
        rho_00 = dm_flat[idx_00]
        rho_01 = dm_flat[idx_01]
        rho_10 = dm_flat[idx_10]
        rho_11 = dm_flat[idx_11]
        
        u_rho_00 = a * rho_00 + b * rho_10
        u_rho_01 = a * rho_01 + b * rho_11
        u_rho_10 = c * rho_00 + d * rho_10
        u_rho_11 = c * rho_01 + d * rho_11
        
        out_flat[idx_00] = u_rho_00 * a_conj + u_rho_01 * b_conj
        out_flat[idx_01] = u_rho_00 * c_conj + u_rho_01 * d_conj
        out_flat[idx_10] = u_rho_10 * a_conj + u_rho_11 * b_conj
        out_flat[idx_11] = u_rho_10 * c_conj + u_rho_11 * d_conj


@cuda.jit(fastmath=True)
def _dm_two_qubit_kernel(
    dm_flat, out_flat,
    matrix_flat, matrix_conj_flat,
    row_mask_0, row_mask_1, col_mask_0, col_mask_1,
    dim_log2, total_size
):
    """
    Apply U * rho * U† for two-qubit gate on density matrix.
    Processes 16 elements (4x4 block) per iteration using bit masking.
    """
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    dim = 1 << dim_log2
    sixteenth_size = total_size >> 4
    
    pos_0 = 0
    pos_1 = 0
    temp = row_mask_0
    while temp > 1:
        temp >>= 1
        pos_0 += 1
    temp = row_mask_1
    while temp > 1:
        temp >>= 1
        pos_1 += 1
    
    if pos_0 > pos_1:
        pos_0, pos_1 = pos_1, pos_0
    
    for i in range(idx, sixteenth_size, stride):
        flat_i = i
        flat_i = (flat_i & ~((1 << pos_0) - 1)) << 1 | (flat_i & ((1 << pos_0) - 1))
        flat_i = (flat_i & ~((1 << pos_1) - 1)) << 1 | (flat_i & ((1 << pos_1) - 1))
        flat_i = (flat_i & ~((1 << (pos_0 + dim_log2)) - 1)) << 1 | (flat_i & ((1 << (pos_0 + dim_log2)) - 1))
        flat_i = (flat_i & ~((1 << (pos_1 + dim_log2)) - 1)) << 1 | (flat_i & ((1 << (pos_1 + dim_log2)) - 1))
        
        row_0 = flat_i >> dim_log2
        col_0 = flat_i & (dim - 1)
        
        rho = cuda.local.array((4, 4), dtype=numba.complex128)
        for ri in range(4):
            row_idx = row_0
            if ri & 1:
                row_idx |= row_mask_1
            if ri & 2:
                row_idx |= row_mask_0
            for ci in range(4):
                col_idx = col_0
                if ci & 1:
                    col_idx |= col_mask_1
                if ci & 2:
                    col_idx |= col_mask_0
                rho[ri, ci] = dm_flat[row_idx * dim + col_idx]
        
        u_rho = cuda.local.array((4, 4), dtype=numba.complex128)
        for ri in range(4):
            for ci in range(4):
                acc = 0j
                for k in range(4):
                    acc += matrix_flat[ri * 4 + k] * rho[k, ci]
                u_rho[ri, ci] = acc
        
        for ri in range(4):
            row_idx = row_0
            if ri & 1:
                row_idx |= row_mask_1
            if ri & 2:
                row_idx |= row_mask_0
            for ci in range(4):
                col_idx = col_0
                if ci & 1:
                    col_idx |= col_mask_1
                if ci & 2:
                    col_idx |= col_mask_0
                acc = 0j
                for k in range(4):
                    acc += u_rho[ri, k] * matrix_conj_flat[ci * 4 + k]
                out_flat[row_idx * dim + col_idx] = acc


_global_dm_executor: OptimizedDensityMatrixExecutor | None = None
_dm_executor_lock = threading.Lock()


def get_dm_executor(qubit_count: int) -> OptimizedDensityMatrixExecutor:
    """Get or create density matrix GPU executor."""
    global _global_dm_executor
    
    with _dm_executor_lock:
        if _global_dm_executor is None or _global_dm_executor.qubit_count != qubit_count:
            _global_dm_executor = OptimizedDensityMatrixExecutor(qubit_count)
        return _global_dm_executor


def apply_dm_operations_optimized(
    density_matrix: np.ndarray,
    qubit_count: int,
    operations: list
) -> np.ndarray:
    """
    Apply operations to density matrix with GPU optimization.
    
    This keeps the density matrix on GPU throughout circuit execution,
    avoiding per-operation host↔device transfers.
    """
    if not _GPU_AVAILABLE or qubit_count < 6 or density_matrix.size < 256:
        return None
    
    executor = get_dm_executor(qubit_count)
    return executor.execute_circuit(density_matrix, operations)
