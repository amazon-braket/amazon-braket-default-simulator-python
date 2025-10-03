#!/usr/bin/env python3
"""
Test script to demonstrate GPU-enabled linalg_utils functionality.

This script shows how to:
1. Check GPU availability
2. Enable/disable GPU usage
3. Compare performance between CPU and GPU implementations
4. Verify correctness of GPU operations
"""

import os
import time
import numpy as np

# Set environment variable to enable GPU (if available)
os.environ["BRAKET_USE_GPU"] = "true"

from src.braket.default_simulator.linalg_utils import (
    is_gpu_available,
    is_gpu_enabled,
    enable_gpu,
    multiply_matrix,
    QuantumGateDispatcher,
)


def create_test_state(n_qubits: int) -> np.ndarray:
    """Create a normalized random quantum state."""
    state_size = 2**n_qubits
    state_flat = np.random.random(state_size) + 1j * np.random.random(state_size)
    state_flat = state_flat / np.linalg.norm(state_flat)
    return state_flat.reshape([2] * n_qubits)


def hadamard_matrix():
    """Return Hadamard gate matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def cnot_matrix():
    """Return CNOT gate matrix."""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def test_gpu_availability():
    """Test GPU availability and configuration."""
    print("=== GPU Availability Test ===")
    print(f"GPU Available: {is_gpu_available()}")
    print(f"GPU Enabled: {is_gpu_enabled()}")
    
    if is_gpu_available():
        print("✓ GPU support detected")
        
        # Test enabling/disabling GPU
        original_state = is_gpu_enabled()
        
        enable_gpu(True)
        print(f"After enable_gpu(True): {is_gpu_enabled()}")
        
        enable_gpu(False)
        print(f"After enable_gpu(False): {is_gpu_enabled()}")
        
        # Restore original state
        enable_gpu(original_state)
        print(f"Restored to: {is_gpu_enabled()}")
    else:
        print("⚠ No GPU support available (CuPy not installed or no CUDA GPU)")
    
    print()


def test_single_qubit_gates():
    """Test single-qubit gate operations."""
    print("=== Single-Qubit Gate Test ===")
    
    n_qubits = 16  # Large enough to trigger GPU usage
    state = create_test_state(n_qubits)
    matrix = hadamard_matrix()
    target = 0
    
    # Test with CPU
    enable_gpu(False)
    dispatcher_cpu = QuantumGateDispatcher(n_qubits)
    print(f"CPU Dispatcher - use_large: {dispatcher_cpu.use_large}, use_gpu: {dispatcher_cpu.use_gpu}")
    
    start_time = time.time()
    result_cpu = multiply_matrix(state.copy(), matrix, (target,), dispatcher=dispatcher_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU execution time: {cpu_time:.4f} seconds")
    
    # Test with GPU (if available)
    if is_gpu_available():
        enable_gpu(True)
        dispatcher_gpu = QuantumGateDispatcher(n_qubits)
        print(f"GPU Dispatcher - use_large: {dispatcher_gpu.use_large}, use_gpu: {dispatcher_gpu.use_gpu}")
        
        start_time = time.time()
        result_gpu = multiply_matrix(state.copy(), matrix, (target,), dispatcher=dispatcher_gpu)
        gpu_time = time.time() - start_time
        
        print(f"GPU execution time: {gpu_time:.4f} seconds")
        
        # Verify results are the same
        if np.allclose(result_cpu, result_gpu, atol=1e-10):
            print("✓ CPU and GPU results match")
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("✗ CPU and GPU results differ!")
            print(f"Max difference: {np.max(np.abs(result_cpu - result_gpu))}")
    else:
        print("Skipping GPU test (not available)")
    
    print()


def test_two_qubit_gates():
    """Test two-qubit gate operations."""
    print("=== Two-Qubit Gate Test ===")
    
    n_qubits = 16
    state = create_test_state(n_qubits)
    matrix = cnot_matrix()
    targets = (0, 1)
    
    # Test with CPU
    enable_gpu(False)
    dispatcher_cpu = QuantumGateDispatcher(n_qubits)
    
    start_time = time.time()
    result_cpu = multiply_matrix(state.copy(), matrix, targets, dispatcher=dispatcher_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU execution time: {cpu_time:.4f} seconds")
    
    # Test with GPU (if available)
    if is_gpu_available():
        enable_gpu(True)
        dispatcher_gpu = QuantumGateDispatcher(n_qubits)
        
        start_time = time.time()
        result_gpu = multiply_matrix(state.copy(), matrix, targets, dispatcher=dispatcher_gpu)
        gpu_time = time.time() - start_time
        
        print(f"GPU execution time: {gpu_time:.4f} seconds")
        
        # Verify results are the same
        if np.allclose(result_cpu, result_gpu, atol=1e-10):
            print("✓ CPU and GPU results match")
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("✗ CPU and GPU results differ!")
            print(f"Max difference: {np.max(np.abs(result_cpu - result_gpu))}")
    else:
        print("Skipping GPU test (not available)")
    
    print()


def test_dispatcher_selection():
    """Test that the dispatcher correctly selects implementations."""
    print("=== Dispatcher Selection Test ===")
    
    # Test small circuit (should use small implementations)
    small_dispatcher = QuantumGateDispatcher(5)
    print(f"5-qubit circuit: use_large={small_dispatcher.use_large}, use_gpu={small_dispatcher.use_gpu}")
    
    # Test medium circuit (should use large CPU implementations)
    medium_dispatcher = QuantumGateDispatcher(12)
    print(f"12-qubit circuit: use_large={medium_dispatcher.use_large}, use_gpu={medium_dispatcher.use_gpu}")
    
    # Test large circuit (should use GPU if available)
    if is_gpu_available():
        enable_gpu(True)
        large_dispatcher = QuantumGateDispatcher(18)
        print(f"18-qubit circuit (GPU enabled): use_large={large_dispatcher.use_large}, use_gpu={large_dispatcher.use_gpu}")
        
        # Test forcing CPU
        large_dispatcher_cpu = QuantumGateDispatcher(18, force_cpu=True)
        print(f"18-qubit circuit (forced CPU): use_large={large_dispatcher_cpu.use_large}, use_gpu={large_dispatcher_cpu.use_gpu}")
    else:
        large_dispatcher = QuantumGateDispatcher(18)
        print(f"18-qubit circuit (no GPU): use_large={large_dispatcher.use_large}, use_gpu={large_dispatcher.use_gpu}")
    
    print()


def benchmark_performance():
    """Benchmark performance across different qubit counts."""
    print("=== Performance Benchmark ===")
    
    qubit_counts = [10, 12, 14, 16, 18]
    matrix = hadamard_matrix()
    
    print("Qubit Count | CPU Time (s) | GPU Time (s) | Speedup")
    print("-" * 55)
    
    for n_qubits in qubit_counts:
        state = create_test_state(n_qubits)
        target = 0
        
        # CPU benchmark
        enable_gpu(False)
        dispatcher_cpu = QuantumGateDispatcher(n_qubits)
        
        start_time = time.time()
        result_cpu = multiply_matrix(state.copy(), matrix, (target,), dispatcher=dispatcher_cpu)
        cpu_time = time.time() - start_time
        
        # GPU benchmark (if available)
        if is_gpu_available():
            enable_gpu(True)
            dispatcher_gpu = QuantumGateDispatcher(n_qubits)
            
            start_time = time.time()
            result_gpu = multiply_matrix(state.copy(), matrix, (target,), dispatcher=dispatcher_gpu)
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            # Verify correctness
            correct = np.allclose(result_cpu, result_gpu, atol=1e-10)
            status = "✓" if correct else "✗"
            
            print(f"{n_qubits:11d} | {cpu_time:12.4f} | {gpu_time:12.4f} | {speedup:7.2f}x {status}")
        else:
            print(f"{n_qubits:11d} | {cpu_time:12.4f} | {'N/A':>12} | {'N/A':>7}")
    
    print()


def main():
    """Run all tests."""
    print("GPU-Enabled Linalg Utils Test Suite")
    print("=" * 40)
    print()
    
    test_gpu_availability()
    test_dispatcher_selection()
    test_single_qubit_gates()
    test_two_qubit_gates()
    
    if is_gpu_available():
        benchmark_performance()
    else:
        print("Skipping performance benchmark (GPU not available)")
    
    print("Test suite completed!")


if __name__ == "__main__":
    main()
