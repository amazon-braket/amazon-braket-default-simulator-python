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
GPU single operation strategy for quantum simulations.

This module provides GPU acceleration for quantum circuit execution using
efficient ping-pong buffering, cuda.jit kernels, and advanced memory management
for scaling to high qubit counts.

OPTIMIZATION NOTES (via gpu_optimized_operations.py):
- Single host→GPU transfer at circuit start, single transfer back at end
- Matrix caching with LRU eviction to avoid repeated uploads
- Pinned memory for faster transfers on large states
- Adaptive thread/block configuration based on problem size
- Operation fusion for consecutive single-qubit gates (threshold: 2)
- Batch phase kernel for consecutive diagonal gates
- Persistent kernel for processing multiple gates in one launch
- In-place operations for diagonal gates (no buffer swap)
- CUDA events for fine-grained synchronization
- Shared memory tiling for better cache utilization
- Multi-stream pipelining for independent qubit operations
- Warp-aligned configuration for coalesced memory access
"""

import numpy as np
from numba import cuda

from braket.default_simulator.gpu_optimized_operations import (
    apply_operations_optimized,
    clear_matrix_cache,
)
from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
    _OPTIMAL_THREADS_PER_BLOCK,
)
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation_strategies import single_operation_strategy


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply operations to state vector using GPU acceleration with optimized memory management.
    
    This function uses the optimized GPU executor which includes:
    - Single host→GPU transfer at start, single transfer back at end
    - Advanced operation fusion (threshold: 2 gates)
    - Batch phase kernel for consecutive diagonal gates (Z, S, T, Rz)
    - In-place diagonal operations (no buffer swap overhead)
    - CUDA events for fine-grained synchronization
    - Warp-aligned config for coalesced memory access on low target qubits
    - Multi-stream pipelining for independent qubit operations
    - Shared memory tiling for matrix and state data
    """
    MIN_GPU_QUBITS = 18
    MIN_GPU_STATE_SIZE = 2**18
    MIN_OPS_FOR_GPU = 50
    
    memory_info = _check_gpu_memory_availability(state.size)
    use_gpu = (
        _GPU_AVAILABLE
        and qubit_count >= MIN_GPU_QUBITS
        and state.size >= MIN_GPU_STATE_SIZE
        and len(operations) >= MIN_OPS_FOR_GPU
        and memory_info['can_fit']
    )
    
    if not use_gpu:
        return single_operation_strategy.apply_operations(state, qubit_count, operations)
    
    return apply_operations_optimized(state, qubit_count, operations)


def clear_gpu_caches():
    """Clear GPU caches."""
    if not _GPU_AVAILABLE:
        return
    
    clear_matrix_cache()


def _check_gpu_memory_availability(state_size: int) -> dict:
    """Check if GPU has sufficient memory for the given state size.
    
    Args:
        state_size: Number of complex128 elements in the state vector
        
    Returns:
        Dictionary with memory availability information
    """
    if not _GPU_AVAILABLE:
        return {'can_fit': False, 'required_gb': 0, 'available_gb': 0}
    
    try:
        free_bytes, total_bytes = cuda.current_context().get_memory_info()
    except Exception:
        return {'can_fit': False, 'required_gb': 0, 'available_gb': 0}
    
    bytes_per_complex = 16
    state_bytes = state_size * bytes_per_complex
    required_bytes = state_bytes * 2.2
    
    safety_margin = total_bytes * 0.10
    effective_available = free_bytes - safety_margin
    can_fit = required_bytes <= effective_available
    
    return {
        'can_fit': can_fit,
        'required_gb': required_bytes / (1024**3),
        'available_gb': effective_available / (1024**3),
        'total_gb': total_bytes / (1024**3),
        'free_gb': free_bytes / (1024**3)
    }


def _get_optimal_launch_config(total_size: int, qubit_count: int) -> tuple[int, int]:
    """Calculate optimal GPU launch configuration."""
    if total_size >= 2**22:
        threads_per_block = 256
    elif total_size >= 2**18:
        threads_per_block = 512
    else:
        threads_per_block = _OPTIMAL_THREADS_PER_BLOCK
    
    blocks_needed = (total_size + threads_per_block - 1) // threads_per_block
    blocks_per_grid = min(blocks_needed, _MAX_BLOCKS_PER_GRID)
    
    return blocks_per_grid, threads_per_block
