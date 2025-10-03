# GPU Support for Amazon Braket Default Simulator

This document describes the GPU acceleration features added to the Amazon Braket Default Simulator's `linalg_utils` module.

## Overview

The `linalg_utils` module now supports conditional GPU acceleration using CuPy for NVIDIA GPUs. The implementation automatically selects between CPU and GPU execution based on:

1. **GPU availability** (CuPy installed and CUDA GPU present)
2. **Circuit size** (GPU threshold: 15+ qubits)
3. **User configuration** (environment variables and API calls)

## Installation

### Basic Installation (CPU only)
```bash
pip install amazon-braket-default-simulator
```

### GPU Installation
```bash
# Install with GPU support
pip install amazon-braket-default-simulator[gpu]

# Or install CuPy separately (for specific CUDA versions)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

## Usage

### Environment Variable Control

The simplest way to enable GPU acceleration:

```bash
# Enable GPU (if available)
export BRAKET_USE_GPU=true

# Disable GPU (force CPU)
export BRAKET_USE_GPU=false
```

### Programmatic Control

```python
from braket.default_simulator.linalg_utils import (
    is_gpu_available,
    is_gpu_enabled,
    enable_gpu
)

# Check GPU availability
print(f"GPU Available: {is_gpu_available()}")
print(f"GPU Enabled: {is_gpu_enabled()}")

# Enable/disable GPU programmatically
enable_gpu(True)   # Enable GPU
enable_gpu(False)  # Disable GPU (force CPU)
```

### Automatic Selection

The system automatically chooses the best implementation:

```python
from braket.default_simulator.linalg_utils import QuantumGateDispatcher

# Small circuits (≤10 qubits): NumPy-based implementations
small_dispatcher = QuantumGateDispatcher(5)
print(f"Small circuit: use_gpu={small_dispatcher.use_gpu}")

# Medium circuits (11-14 qubits): Numba JIT-compiled implementations  
medium_dispatcher = QuantumGateDispatcher(12)
print(f"Medium circuit: use_gpu={medium_dispatcher.use_gpu}")

# Large circuits (≥15 qubits): GPU implementations (if available)
large_dispatcher = QuantumGateDispatcher(18)
print(f"Large circuit: use_gpu={large_dispatcher.use_gpu}")

# Force CPU usage even for large circuits
cpu_dispatcher = QuantumGateDispatcher(18, force_cpu=True)
print(f"Forced CPU: use_gpu={cpu_dispatcher.use_gpu}")
```

## Supported Operations

The following quantum gate operations support GPU acceleration:

### Single-Qubit Gates
- General single-qubit gates (Hadamard, Pauli-X, Y, rotations, etc.)
- Diagonal gates (Pauli-Z, S, T, RZ, Phase shift) - optimized implementation

### Two-Qubit Gates
- CNOT gates - optimized implementation
- SWAP gates - optimized implementation
- General two-qubit gates (CZ, iSWAP, etc.)
- Controlled phase shift gates - optimized implementation

### Multi-Qubit Gates
- General n-qubit gates (fallback to tensordot operations)

## Performance Characteristics

### GPU Threshold
- **Small circuits** (≤10 qubits): CPU implementations are faster due to low overhead
- **Medium circuits** (11-14 qubits): Numba JIT-compiled CPU implementations
- **Large circuits** (≥15 qubits): GPU implementations provide speedup

### Memory Considerations
- GPU memory usage scales as 2^n for n-qubit states
- Automatic fallback to CPU if GPU memory is insufficient
- Data transfer overhead between CPU and GPU memory

### Expected Speedups
Typical speedups for large circuits (≥16 qubits):
- Single-qubit gates: 2-5x speedup
- Two-qubit gates: 3-8x speedup
- Performance varies by GPU model and circuit complexity

## Testing

### Run the Test Suite
```bash
python test_gpu_linalg.py
```

This will:
1. Check GPU availability
2. Test dispatcher selection logic
3. Verify correctness of GPU implementations
4. Benchmark performance across different circuit sizes

### Example Output
```
=== GPU Availability Test ===
GPU Available: True
GPU Enabled: True
✓ GPU support detected

=== Single-Qubit Gate Test ===
CPU Dispatcher - use_large: True, use_gpu: False
CPU execution time: 0.0234 seconds
GPU Dispatcher - use_large: True, use_gpu: True
GPU execution time: 0.0089 seconds
✓ CPU and GPU results match
Speedup: 2.63x
```

## Troubleshooting

### Common Issues

1. **CuPy Import Error**
   ```
   ImportError: No module named 'cupy'
   ```
   **Solution**: Install CuPy with `pip install cupy-cuda12x`

2. **CUDA Not Available**
   ```
   GPU Available: False
   ```
   **Solution**: Ensure NVIDIA GPU drivers and CUDA toolkit are installed

3. **GPU Memory Error**
   ```
   cupy.cuda.memory.OutOfMemoryError
   ```
   **Solution**: Reduce circuit size or use `force_cpu=True`

### Debug Information

Enable debug output:
```python
import os
os.environ["BRAKET_USE_GPU"] = "true"

# Import will show GPU status
from braket.default_simulator.linalg_utils import is_gpu_available
```

## Implementation Details

### Architecture
- **Conditional imports**: CuPy is imported only if available
- **Fallback mechanisms**: Automatic CPU fallback if GPU operations fail
- **Memory management**: Efficient CPU↔GPU data transfers
- **Type preservation**: Results maintain original NumPy array types

### GPU Functions
Each CPU implementation has a corresponding GPU version:
- `_apply_single_qubit_gate_gpu()`
- `_apply_diagonal_gate_gpu()`
- `_apply_cnot_gpu()`
- `_apply_swap_gpu()`
- `_apply_controlled_phase_shift_gpu()`
- `_apply_two_qubit_gate_gpu()`

### Dispatcher Logic
```python
class QuantumGateDispatcher:
    def __init__(self, n_qubits: int, force_cpu: bool = False):
        self.use_gpu = (_USE_GPU and n_qubits > _GPU_QUBIT_THRESHOLD and not force_cpu)
        
        if self.use_gpu:
            # Assign GPU implementations
        elif n_qubits > _QUBIT_THRESHOLD:
            # Assign Numba implementations  
        else:
            # Assign NumPy implementations
```

## Configuration Reference

### Environment Variables
- `BRAKET_USE_GPU`: Enable/disable GPU usage (`"true"`, `"1"`, `"yes"` to enable)

### Thresholds
- `_QUBIT_THRESHOLD = 10`: CPU small→large implementation threshold
- `_GPU_QUBIT_THRESHOLD = 15`: CPU→GPU implementation threshold

### API Functions
- `is_gpu_available()`: Check if GPU support is available
- `is_gpu_enabled()`: Check if GPU usage is currently enabled
- `enable_gpu(enable: bool)`: Enable/disable GPU usage programmatically

## Future Enhancements

Potential improvements for future versions:
1. **Multi-GPU support**: Distribute computation across multiple GPUs
2. **Memory optimization**: Streaming for very large circuits
3. **Custom kernels**: Optimized CUDA kernels for specific gate types
4. **AMD GPU support**: ROCm/HIP support for AMD GPUs
5. **Automatic tuning**: Dynamic threshold adjustment based on hardware
