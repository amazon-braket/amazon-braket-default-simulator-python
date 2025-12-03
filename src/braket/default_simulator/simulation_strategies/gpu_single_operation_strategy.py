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

OPTIMIZATION NOTES:
- Uses single host→GPU transfer at circuit start
- Uses single GPU→host transfer at circuit end
- Matrix caching eliminates repeated uploads for common gates
- Minimal synchronization points for better throughput
"""

import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
    _OPTIMAL_THREADS_PER_BLOCK,
)
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation_strategies import single_operation_strategy

from braket.default_simulator.gpu_optimized_operations import (
    apply_operations_optimized,
    clear_matrix_cache,
)


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Apply operations to state vector using GPU acceleration with optimized memory management.
    
    This function uses the optimized GPU executor which:
    - Performs a single host→GPU transfer at the start
    - Executes all operations on GPU without intermediate transfers
    - Performs a single GPU→host transfer at the end
    - Caches gate matrices to avoid repeated uploads
    """
    memory_info = _check_gpu_memory_availability(state.size)
    use_gpu = (_GPU_AVAILABLE and 
               qubit_count >= 8 and 
               state.size >= 256 and 
               memory_info['can_fit'])
    
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
