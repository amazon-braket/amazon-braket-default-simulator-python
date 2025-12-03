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
Persistent GPU state manager for quantum simulations.

This module provides advanced GPU state management that eliminates redundant
host↔device transfers through intelligent caching, zero-copy optimization,
and unified memory when available.
"""

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np
from numba import cuda

from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
    _OPTIMAL_THREADS_PER_BLOCK,
)


class GPUMemoryPool:
    """GPU memory pool for efficient allocation and deallocation with zero-copy optimization."""
    
    def __init__(self, initial_pool_size_mb: int = 512):
        self.free_buffers: dict[tuple[int, np.dtype], list[cuda.devicearray.DeviceNDArray]] = {}
        self.allocated_buffers: dict[int, cuda.devicearray.DeviceNDArray] = {}
        self.buffer_sizes: dict[int, tuple[int, np.dtype]] = {}
        self.allocation_times: dict[int, float] = {}
        self.reference_counts: dict[int, int] = {}
        self.lock = threading.RLock()
        self.total_allocated = 0
        self.peak_allocated = 0
        self.initial_pool_size = initial_pool_size_mb * 1024 * 1024
        self.unified_memory_enabled = self._check_unified_memory()
        self.pinned_host_buffers: dict[int, np.ndarray] = {}
        
        self._preallocate_common_sizes()
    
    def _check_unified_memory(self) -> bool:
        """Check if CUDA unified memory is available."""
        try:
            if not _GPU_AVAILABLE:
                return False
            
            test_size = 1024
            try:
                ptr = cuda.cuda.cuMemAllocManaged(test_size, cuda.cuda.CU_MEM_ATTACH_GLOBAL)[1]
                cuda.cuda.cuMemFree(ptr)
                return True
            except:
                return False
        except Exception:
            return False
    
    def _preallocate_common_sizes(self):
        """Pre-allocate common quantum state vector sizes."""
        if not _GPU_AVAILABLE:
            return
            
        common_qubit_counts = [10, 12, 14, 16, 18, 20]
        dtype = np.complex128
        
        with self.lock:
            for qubits in common_qubit_counts:
                size = 1 << qubits
                buffer_size = size * np.dtype(dtype).itemsize
                
                if buffer_size <= self.initial_pool_size // len(common_qubit_counts):
                    try:
                        if self.unified_memory_enabled:
                            buffer = self._allocate_unified_memory_buffer(size, dtype)
                        else:
                            buffer = cuda.device_array(size, dtype=dtype)
                            
                        key = (size, dtype)
                        if key not in self.free_buffers:
                            self.free_buffers[key] = []
                        self.free_buffers[key].append(buffer)
                        
                    except Exception as e:
                        print(f"Warning: Failed to preallocate buffer for {qubits} qubits: {e}")
    
    def _allocate_unified_memory_buffer(self, size: int, dtype: np.dtype) -> cuda.devicearray.DeviceNDArray:
        """Allocate CUDA unified memory buffer for zero-copy access."""
        try:
            itemsize = np.dtype(dtype).itemsize
            total_bytes = size * itemsize
            
            ptr, _ = cuda.cuda.cuMemAllocManaged(total_bytes, cuda.cuda.CU_MEM_ATTACH_GLOBAL)
            
            buffer = cuda.devicearray.DeviceNDArray(
                shape=(size,),
                strides=(itemsize,),
                dtype=dtype,
                gpu_data=numba.cuda.cudadrv.driver.MemoryPointer(
                    numba.cuda.current_context(), ptr, total_bytes
                )
            )
            
            return buffer
            
        except Exception as e:
            return cuda.device_array(size, dtype=dtype)
    
    def allocate(self, size: int, dtype: np.dtype = np.complex128, 
                 shape: tuple[int, ...] | None = None) -> tuple[cuda.devicearray.DeviceNDArray, int]:
        """Allocate GPU buffer with optional zero-copy capability."""
        if shape is None:
            shape = (size,)
        
        key = (size, dtype)
        
        with self.lock:
            if key in self.free_buffers and self.free_buffers[key]:
                buffer = self.free_buffers[key].pop()
                buffer_id = id(buffer)
            else:
                if self.unified_memory_enabled:
                    buffer = self._allocate_unified_memory_buffer(size, dtype)
                else:
                    buffer = cuda.device_array(size, dtype=dtype)
                    host_buffer = cuda.pinned_array(size, dtype=dtype)
                    self.pinned_host_buffers[id(buffer)] = host_buffer
                
                buffer_id = id(buffer)
            
            if buffer.shape != shape:
                buffer = buffer.reshape(shape)
            
            self.allocated_buffers[buffer_id] = buffer
            self.buffer_sizes[buffer_id] = key
            self.allocation_times[buffer_id] = time.time()
            self.reference_counts[buffer_id] = 1
            self.total_allocated += size * np.dtype(dtype).itemsize
            self.peak_allocated = max(self.peak_allocated, self.total_allocated)
            
            return buffer, buffer_id
    
    def deallocate(self, buffer_id: int):
        """Deallocate GPU buffer and return to pool."""
        with self.lock:
            if buffer_id not in self.allocated_buffers:
                return
            
            buffer = self.allocated_buffers[buffer_id]
            key = self.buffer_sizes[buffer_id]
            
            self.reference_counts[buffer_id] -= 1
            
            if self.reference_counts[buffer_id] <= 0:
                if key not in self.free_buffers:
                    self.free_buffers[key] = []
                self.free_buffers[key].append(buffer)
                
                del self.allocated_buffers[buffer_id]
                del self.buffer_sizes[buffer_id]
                del self.allocation_times[buffer_id]
                del self.reference_counts[buffer_id]
                
                if buffer_id in self.pinned_host_buffers:
                    del self.pinned_host_buffers[buffer_id]
                
                self.total_allocated -= key[0] * np.dtype(key[1]).itemsize
    
    def add_reference(self, buffer_id: int):
        """Increment reference count for a buffer."""
        with self.lock:
            if buffer_id in self.reference_counts:
                self.reference_counts[buffer_id] += 1
    
    def get_pinned_host_buffer(self, buffer_id: int) -> np.ndarray | None:
        """Get associated pinned host buffer for faster transfers."""
        return self.pinned_host_buffers.get(buffer_id)
    
    def cleanup_old_buffers(self, max_age_seconds: float = 300.0):
        """Clean up old unused buffers to free memory."""
        current_time = time.time()
        
        with self.lock:
            to_remove = []
            for buffer_id, alloc_time in self.allocation_times.items():
                if (current_time - alloc_time > max_age_seconds and 
                    self.reference_counts.get(buffer_id, 0) <= 0):
                    to_remove.append(buffer_id)
            
            for buffer_id in to_remove:
                self.deallocate(buffer_id)
    
    def force_cleanup(self):
        """Force immediate cleanup of all GPU buffers."""
        with self.lock:
            for buffer_list in self.free_buffers.values():
                buffer_list.clear()
            self.free_buffers.clear()
            
            buffer_ids = list(self.allocated_buffers.keys())
            for buffer_id in buffer_ids:
                self.deallocate(buffer_id)
            
            self.allocated_buffers.clear()
            self.buffer_sizes.clear()
            self.allocation_times.clear()
            self.reference_counts.clear()
            self.pinned_host_buffers.clear()
            
            self.total_allocated = 0
            self.peak_allocated = 0


class PersistentGPUState:
    """Represents a persistent quantum state vector on GPU with zero-copy optimization."""
    
    def __init__(self, buffer: cuda.devicearray.DeviceNDArray, buffer_id: int, 
                 memory_pool: GPUMemoryPool, host_data: np.ndarray | None = None):
        self.buffer = buffer
        self.buffer_id = buffer_id
        self.memory_pool = memory_pool
        self.host_data = host_data
        self.last_access_time = time.time()
        self.access_count = 0
        self.is_dirty = False
        
        self._finalizer = weakref.finalize(self, self._cleanup, buffer_id, memory_pool)
    
    @staticmethod
    def _cleanup(buffer_id: int, memory_pool: GPUMemoryPool):
        """Cleanup method called when object is garbage collected."""
        memory_pool.deallocate(buffer_id)
    
    def access(self):
        """Mark this state as accessed."""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def get_host_view(self) -> np.ndarray:
        """Get zero-copy host view if using unified memory, otherwise copy to pinned buffer."""
        self.access()
        
        if self.host_data is not None:
            return self.host_data
        else:
            pinned_buffer = self.memory_pool.get_pinned_host_buffer(self.buffer_id)
            if pinned_buffer is not None:
                self.buffer.copy_to_host(ary=pinned_buffer)
                return pinned_buffer
            else:
                return self.buffer.copy_to_host()
    
    def sync_from_host(self, host_array: np.ndarray):
        """Synchronize GPU state from host array."""
        self.access()
        self.is_dirty = True
        
        if self.host_data is not None:
            np.copyto(self.host_data, host_array)
        else:
            cuda.to_device(host_array, to=self.buffer)


class PersistentGPUStateManager:
    """Advanced GPU state manager that eliminates redundant host↔device transfers.
    
    Key optimizations:
    - Caches GPU buffers by shape/dtype to avoid repeated allocations
    - Uses pinned memory for faster host↔device transfers
    - Implements LRU eviction for memory management
    - Provides zero-copy access when unified memory is available
    """
    
    def __init__(self, max_cached_states: int = 16, cleanup_interval_seconds: float = 120.0):
        self.memory_pool = GPUMemoryPool()
        self.persistent_states: dict[tuple[tuple[int, ...], np.dtype], PersistentGPUState] = {}
        self.state_access_order: list[tuple[tuple[int, ...], np.dtype]] = []
        self.max_cached_states = max_cached_states
        self.cleanup_interval = cleanup_interval_seconds
        self.last_cleanup = time.time()
        self.lock = threading.RLock()
        self._shutdown_requested = False
        
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-cleanup")
        self._schedule_cleanup()
    
    def _schedule_cleanup(self):
        """Schedule periodic cleanup of old GPU states."""
        def cleanup_task():
            if self._shutdown_requested:
                return
            time.sleep(self.cleanup_interval)
            if not self._shutdown_requested:
                self._cleanup_old_states()
                self._schedule_cleanup()
        
        if not self._shutdown_requested:
            self.cleanup_executor.submit(cleanup_task)
    
    def get_persistent_state(self, host_state: np.ndarray, 
                           force_refresh: bool = False) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get persistent GPU state, avoiding transfer if already cached."""
        key = (host_state.shape, host_state.dtype)
        
        with self.lock:
            if not force_refresh and key in self.persistent_states:
                persistent_state = self.persistent_states[key]
                persistent_state.access()
                
                if key in self.state_access_order:
                    self.state_access_order.remove(key)
                self.state_access_order.append(key)
                
                return self._get_ping_pong_buffers(persistent_state.buffer)
            
            buffer_a, buffer_a_id = self.memory_pool.allocate(host_state.size, host_state.dtype, host_state.shape)
            buffer_b, buffer_b_id = self.memory_pool.allocate(host_state.size, host_state.dtype, host_state.shape)
            
            if self.memory_pool.unified_memory_enabled:
                host_view = buffer_a.view()
                np.copyto(host_view, host_state)
                persistent_state = PersistentGPUState(buffer_a, buffer_a_id, self.memory_pool, host_view)
            else:
                cuda.to_device(host_state, to=buffer_a)
                persistent_state = PersistentGPUState(buffer_a, buffer_a_id, self.memory_pool)
            
            self.persistent_states[key] = persistent_state
            self.state_access_order.append(key)
            
            self._evict_old_states()
            
            return buffer_a, buffer_b
    
    def _get_ping_pong_buffers(self, primary_buffer: cuda.devicearray.DeviceNDArray) -> tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """Get ping-pong buffers for the given primary buffer."""
        buffer_b, _ = self.memory_pool.allocate(primary_buffer.size, primary_buffer.dtype, primary_buffer.shape)
        return primary_buffer, buffer_b
    
    def _evict_old_states(self):
        """Evict least recently used states when cache is full."""
        while len(self.persistent_states) > self.max_cached_states:
            oldest_key = self.state_access_order.pop(0)
            if oldest_key in self.persistent_states:
                old_state = self.persistent_states[oldest_key]
                del self.persistent_states[oldest_key]
    
    def _cleanup_old_states(self):
        """Periodic cleanup of old GPU memory."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            self.memory_pool.cleanup_old_buffers(max_age_seconds=300.0)
            
            old_keys = []
            for key, state in self.persistent_states.items():
                if current_time - state.last_access_time > 600.0:
                    old_keys.append(key)
            
            for key in old_keys:
                if key in self.persistent_states:
                    del self.persistent_states[key]
                if key in self.state_access_order:
                    self.state_access_order.remove(key)
            
            self.last_cleanup = current_time
            
            gc.collect()
    
    def get_result_array(self, gpu_buffer: cuda.devicearray.DeviceNDArray, 
                        use_zero_copy: bool = True) -> np.ndarray:
        """Get result array from GPU buffer with zero-copy optimization when possible."""
        if use_zero_copy and self.memory_pool.unified_memory_enabled:
            return gpu_buffer.view()
        else:
            buffer_id = id(gpu_buffer)
            pinned_buffer = self.memory_pool.get_pinned_host_buffer(buffer_id)
            
            if pinned_buffer is not None:
                try:
                    if pinned_buffer.shape != gpu_buffer.shape:
                        pinned_reshaped = pinned_buffer.reshape(gpu_buffer.shape)
                        gpu_buffer.copy_to_host(ary=pinned_reshaped)
                        return pinned_reshaped.copy()
                    else:
                        gpu_buffer.copy_to_host(ary=pinned_buffer)
                        return pinned_buffer.copy()
                except (ValueError, RuntimeError):
                    return gpu_buffer.copy_to_host()
            else:
                return gpu_buffer.copy_to_host()
    
    def clear_cache(self):
        """Clear all cached GPU states."""
        with self.lock:
            self.persistent_states.clear()
            self.state_access_order.clear()
    
    def force_cleanup(self):
        """Force immediate cleanup of all GPU resources."""
        with self.lock:
            self.persistent_states.clear()
            self.state_access_order.clear()
            
            self.memory_pool.force_cleanup()
            
            import gc
            gc.collect()
    
    def shutdown(self):
        """Shutdown the manager and clean up resources."""
        self._shutdown_requested = True
        self.clear_cache()
        self.cleanup_executor.shutdown(wait=False)


# Global instance
_persistent_gpu_manager = None

def get_persistent_gpu_manager() -> PersistentGPUStateManager | None:
    """Get global persistent GPU state manager instance."""
    global _persistent_gpu_manager
    
    if not _GPU_AVAILABLE:
        return None
    
    if _persistent_gpu_manager is None:
        _persistent_gpu_manager = PersistentGPUStateManager()
    
    return _persistent_gpu_manager
