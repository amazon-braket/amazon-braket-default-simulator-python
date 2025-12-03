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

This module consolidates GPU operations to eliminate redundant memory transfers
by keeping state on GPU throughout circuit execution and using efficient caching
for matrices and intermediate results.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    DIAGONAL_GATES,
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
)

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation


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
    3. Batched kernel launches where possible
    4. Minimal synchronization points
    """

    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.matrix_cache = GPUMatrixCache()
        self._stream = None

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

        state_contiguous = np.ascontiguousarray(state)
        gpu_state = cuda.to_device(state_contiguous, stream=self._stream)
        gpu_temp = cuda.device_array_like(gpu_state, stream=self._stream)
        self._stream.synchronize()

        current_buffer = gpu_state
        temp_buffer = gpu_temp

        for op in operations:
            needs_swap = self._apply_operation(op, current_buffer, temp_buffer)
            if needs_swap:
                current_buffer, temp_buffer = temp_buffer, current_buffer

        self._stream.synchronize()
        result = current_buffer.copy_to_host()

        return result

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
            if gate_type == "cx":
                return self._apply_cnot(
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
        """Apply single qubit gate with optimized kernel."""
        n_qubits = len(state_gpu.shape)
        target_bit = n_qubits - target - 1
        half_size = state_gpu.size >> 1

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        threads = 512
        blocks = min((half_size + threads - 1) // threads, _MAX_BLOCKS_PER_GRID)

        if gate_type and gate_type in DIAGONAL_GATES:
            a, d = matrix[0, 0], matrix[1, 1]
            target_mask = 1 << target_bit
            _diagonal_kernel[blocks, threads, self._stream](
                state_flat, out_flat, a, d, target_mask, state_gpu.size
            )
        else:
            a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
            mask = (1 << target_bit) - 1
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

        threads = 512
        blocks = min((total_size + threads - 1) // threads, _MAX_BLOCKS_PER_GRID)

        _cnot_kernel[blocks, threads, self._stream](
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

        threads = 512
        blocks = min((iterations + threads - 1) // threads, _MAX_BLOCKS_PER_GRID)

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
        mask_0 = 1 << (n_qubits - 1 - target0)
        mask_1 = 1 << (n_qubits - 1 - target1)
        mask_both = mask_0 | mask_1
        total_size = state_gpu.size

        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)

        cache_key = f"2q_{hash(matrix.tobytes())}"
        matrix_gpu = self.matrix_cache.get_or_upload(matrix, cache_key)

        threads = 512
        blocks = min((total_size + threads - 1) // threads, _MAX_BLOCKS_PER_GRID)

        _two_qubit_kernel[blocks, threads, self._stream](
            state_flat, out_flat, matrix_gpu, mask_0, mask_1, mask_both, total_size
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

        threads = 512
        blocks = min((total_size + threads - 1) // threads, _MAX_BLOCKS_PER_GRID)

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
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, half_size, stride):
        idx0 = ((i >> n) << (n + 1)) | (i & mask)
        idx1 = idx0 | (1 << n)

        s0 = state_flat[idx0]
        s1 = state_flat[idx1]

        out_flat[idx0] = a * s0 + b * s1
        out_flat[idx1] = c * s0 + d * s1


@cuda.jit(fastmath=True)
def _diagonal_kernel(state_flat, out_flat, a, d, target_mask, total_size):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, total_size, stride):
        factor = d if (i & target_mask) else a
        out_flat[i] = factor * state_flat[i]


@cuda.jit(fastmath=True)
def _cnot_kernel(state_flat, out_flat, control_mask, target_mask, total_size):
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
    state_flat, out_flat, matrix_flat, mask_0, mask_1, mask_both, total_size
):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, total_size, stride):
        if (i & mask_both) == 0:
            s0 = state_flat[i]
            s1 = state_flat[i | mask_1]
            s2 = state_flat[i | mask_0]
            s3 = state_flat[i | mask_both]

            out_flat[i] = (
                matrix_flat[0] * s0
                + matrix_flat[1] * s1
                + matrix_flat[2] * s2
                + matrix_flat[3] * s3
            )
            out_flat[i | mask_1] = (
                matrix_flat[4] * s0
                + matrix_flat[5] * s1
                + matrix_flat[6] * s2
                + matrix_flat[7] * s3
            )
            out_flat[i | mask_0] = (
                matrix_flat[8] * s0
                + matrix_flat[9] * s1
                + matrix_flat[10] * s2
                + matrix_flat[11] * s3
            )
            out_flat[i | mask_both] = (
                matrix_flat[12] * s0
                + matrix_flat[13] * s1
                + matrix_flat[14] * s2
                + matrix_flat[15] * s3
            )


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
