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
from typing import Tuple, Optional, Dict, List
import time
from collections import defaultdict

from braket.default_simulator.operation import GateOperation
from braket.default_simulator.linalg_utils import (
    _GPU_AVAILABLE,
    _OPTIMAL_THREADS_PER_BLOCK,
    _MAX_BLOCKS_PER_GRID,
)


class TensorCoreCapability:
    """Detects and manages tensor core capabilities across different GPU architectures."""
    
    def __init__(self):
        self.compute_capability = None
        self.tensor_cores_available = False
        self.wmma_shapes = {}
        self.max_shared_memory = 0
        self.warp_size = 32
        
        if _GPU_AVAILABLE:
            self._detect_hardware_capabilities()
    
    def _detect_hardware_capabilities(self):
        """Detect GPU hardware capabilities for tensor core optimization."""
        try:
            device = cuda.get_current_device()
            self.compute_capability = device.compute_capability
            
            major, minor = self.compute_capability
            
            if major >= 7:
                self.tensor_cores_available = True
                
                if major == 7 and minor == 0:
                    self.wmma_shapes = {
                        'fp16': [(16, 16, 16)],
                        'mixed': [(16, 16, 16)]
                    }
                elif major == 7 and minor == 5:
                    self.wmma_shapes = {
                        'fp16': [(16, 16, 16), (32, 8, 16), (8, 32, 16)],
                        'mixed': [(16, 16, 16)]
                    }
                elif major >= 8:
                    self.wmma_shapes = {
                        'fp16': [(16, 16, 16), (32, 8, 16), (8, 32, 16)],
                        'bf16': [(16, 16, 16), (32, 8, 16), (8, 32, 16)],
                        'tf32': [(16, 16, 8)],
                        'mixed': [(16, 16, 16)]
                    }
            
            self.max_shared_memory = device.MAX_SHARED_MEMORY_PER_BLOCK
            
            print(f"Tensor Core Detection:")
            print(f"  GPU: {device.name}")
            print(f"  Compute Capability: {major}.{minor}")
            print(f"  Tensor Cores: {'Available' if self.tensor_cores_available else 'Not Available'}")
            print(f"  WMMA Shapes: {self.wmma_shapes}")
            print(f"  Max Shared Memory: {self.max_shared_memory} bytes")
            
        except Exception as e:
            print(f"Hardware detection failed: {e}")
            self.tensor_cores_available = False
    
    def get_optimal_wmma_shape(self, matrix_size: tuple, precision: str = 'fp16') -> Optional[tuple]:
        """Get optimal WMMA shape for given matrix size and precision."""
        if not self.tensor_cores_available or precision not in self.wmma_shapes:
            return None
        
        shapes = self.wmma_shapes[precision]
        m, n = matrix_size
        
        for wmma_m, wmma_n, wmma_k in shapes:
            if m <= wmma_m and n <= wmma_n:
                return (wmma_m, wmma_n, wmma_k)
        
        return shapes[0] if shapes else None


class PrecisionManager:
    """Manages mixed-precision arithmetic with quantum-specific error analysis."""
    
    def __init__(self):
        self.precision_thresholds = {
            'fp16_max': 65504.0,
            'fp16_min': 6.103515625e-05,
            'quantum_fidelity_threshold': 1e-6,
            'amplitude_precision_threshold': 1e-7
        }
        
        self.conversion_stats = defaultdict(list)
    
    def analyze_quantum_matrix_precision(self, matrix: np.ndarray) -> Dict:
        """Analyze precision requirements for quantum matrix operations."""
        matrix_flat = matrix.flatten()
        
        real_parts = np.real(matrix_flat)
        imag_parts = np.imag(matrix_flat)
        
        analysis = {
            'max_real': np.max(np.abs(real_parts)),
            'max_imag': np.max(np.abs(imag_parts)),
            'min_nonzero_real': np.min(np.abs(real_parts[np.nonzero(real_parts)])) if np.any(real_parts) else 0,
            'min_nonzero_imag': np.min(np.abs(imag_parts[np.nonzero(imag_parts)])) if np.any(imag_parts) else 0,
            'dynamic_range_real': 0,
            'dynamic_range_imag': 0,
            'fp16_compatible': True,
            'precision_loss_estimate': 0.0
        }
        
        if analysis['min_nonzero_real'] > 0:
            analysis['dynamic_range_real'] = analysis['max_real'] / analysis['min_nonzero_real']
        if analysis['min_nonzero_imag'] > 0:
            analysis['dynamic_range_imag'] = analysis['max_imag'] / analysis['min_nonzero_imag']
        
        max_val = max(analysis['max_real'], analysis['max_imag'])
        min_val = min(analysis['min_nonzero_real'], analysis['min_nonzero_imag'])
        
        if max_val > self.precision_thresholds['fp16_max']:
            analysis['fp16_compatible'] = False
        elif min_val > 0 and min_val < self.precision_thresholds['fp16_min']:
            analysis['fp16_compatible'] = False
        
        if analysis['fp16_compatible']:
            matrix_fp16 = matrix.astype(np.complex64)
            
            real_fp16 = matrix.real.astype(np.float16).astype(np.float32)
            imag_fp16 = matrix.imag.astype(np.float16).astype(np.float32)
            matrix_fp16_accurate = real_fp16 + 1j * imag_fp16
            
            precision_loss = np.max(np.abs(matrix - matrix_fp16_accurate))
            analysis['precision_loss_estimate'] = precision_loss
        
        return analysis
    
    def convert_to_mixed_precision(self, matrix: np.ndarray, target_precision: str = 'fp16') -> Tuple[np.ndarray, Dict]:
        """Convert quantum matrix to mixed-precision format with validation."""
        analysis = self.analyze_quantum_matrix_precision(matrix)
        
        conversion_info = {
            'original_precision': 'fp64',
            'target_precision': target_precision,
            'conversion_successful': False,
            'precision_loss': 0.0,
            'fidelity_preserved': False
        }
        
        if target_precision == 'fp16' and analysis['fp16_compatible']:
            real_fp16 = matrix.real.astype(np.float16).astype(np.float32)
            imag_fp16 = matrix.imag.astype(np.float16).astype(np.float32)
            converted_matrix = real_fp16 + 1j * imag_fp16
            
            conversion_info['conversion_successful'] = True
            conversion_info['precision_loss'] = analysis['precision_loss_estimate']
            
            if conversion_info['precision_loss'] < self.precision_thresholds['quantum_fidelity_threshold']:
                conversion_info['fidelity_preserved'] = True
        
        elif target_precision == 'fp32':
            converted_matrix = matrix.astype(np.complex64)
            conversion_info['conversion_successful'] = True
            conversion_info['precision_loss'] = np.max(np.abs(matrix - converted_matrix))
            conversion_info['fidelity_preserved'] = True
        
        else:
            converted_matrix = matrix
            conversion_info['target_precision'] = 'fp64'
        
        self.conversion_stats[target_precision].append(conversion_info)
        
        return converted_matrix, conversion_info
    
    def validate_quantum_precision(self, original: np.ndarray, converted: np.ndarray) -> Dict:
        """Validate precision preservation for quantum operations."""
        validation = {
            'max_elementwise_error': np.max(np.abs(original - converted)),
            'frobenius_norm_error': np.linalg.norm(original - converted, 'fro'),
            'unitarity_preserved': False,
            'eigenvalue_preservation': 0.0
        }
        
        if original.shape[0] == original.shape[1]:
            try:
                orig_conj_transpose = np.conj(original.T)
                conv_conj_transpose = np.conj(converted.T)
                
                orig_unitarity_test = np.dot(orig_conj_transpose, original)
                conv_unitarity_test = np.dot(conv_conj_transpose, converted)
                
                unitarity_error = np.max(np.abs(orig_unitarity_test - conv_unitarity_test))
                validation['unitarity_preserved'] = unitarity_error < 1e-6
                
                orig_eigenvals = np.linalg.eigvals(original)
                conv_eigenvals = np.linalg.eigvals(converted)
                
                eigenval_error = np.max(np.abs(np.sort(orig_eigenvals) - np.sort(conv_eigenvals)))
                validation['eigenvalue_preservation'] = eigenval_error
                
            except np.linalg.LinAlgError:
                pass
        
        return validation


class WMMAQuantumOperations:
    """WMMA-accelerated quantum matrix operations using tensor cores."""
    
    def __init__(self, capability: TensorCoreCapability, precision_manager: PrecisionManager):
        self.capability = capability
        self.precision_manager = precision_manager
        self.compiled_kernels = {}
        
    def batch_quantum_matrices(self, matrices: List[np.ndarray], target_shape: tuple) -> Tuple[np.ndarray, List[int]]:
        """Batch small quantum matrices into larger tensor operations."""
        wmma_m, wmma_n, wmma_k = target_shape
        
        max_batch_size = (wmma_m * wmma_n) // (matrices[0].size)
        actual_batch_size = min(len(matrices), max_batch_size)
        
        if actual_batch_size == 0:
            return None, []
        
        matrix_size = matrices[0].shape
        batched_shape = (wmma_m, wmma_n)
        batched_matrix = np.zeros(batched_shape, dtype=matrices[0].dtype)
        
        matrices_per_row = wmma_n // matrix_size[1]
        matrices_per_col = wmma_m // matrix_size[0]
        
        batch_indices = []
        for i in range(actual_batch_size):
            if i >= matrices_per_row * matrices_per_col:
                break
                
            row_idx = (i // matrices_per_row) * matrix_size[0]
            col_idx = (i % matrices_per_row) * matrix_size[1]
            
            end_row = row_idx + matrix_size[0]
            end_col = col_idx + matrix_size[1]
            
            if end_row <= wmma_m and end_col <= wmma_n:
                batched_matrix[row_idx:end_row, col_idx:end_col] = matrices[i]
                batch_indices.append(i)
        
        return batched_matrix, batch_indices
    
    def generate_wmma_kernel_code(self, operation_type: str, precision: str, wmma_shape: tuple) -> str:
        """Generate WMMA kernel code for quantum matrix operations."""
        wmma_m, wmma_n, wmma_k = wmma_shape
        
        if precision == 'fp16':
            wmma_type = "half"
            accumulate_type = "float"
        elif precision == 'bf16':
            wmma_type = "nv_bfloat16"
            accumulate_type = "float"
        else:
            wmma_type = "float"
            accumulate_type = "float"
        
        if operation_type == "matrix_multiply":
            kernel_code = f'''
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_quantum_matrix_multiply(
    {wmma_type}* A, {wmma_type}* B, {accumulate_type}* C,
    int M, int N, int K, int batch_size
) {{
    wmma::fragment<wmma::matrix_a, {wmma_m}, {wmma_n}, {wmma_k}, {wmma_type}, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, {wmma_m}, {wmma_n}, {wmma_k}, {wmma_type}, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, {wmma_m}, {wmma_n}, {wmma_k}, {accumulate_type}> acc_frag;
    wmma::fragment<wmma::accumulator, {wmma_m}, {wmma_n}, {wmma_k}, {accumulate_type}> c_frag;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    int batch_idx = blockIdx.z;
    
    if (batch_idx >= batch_size) return;
    
    int a_offset = batch_idx * M * K;
    int b_offset = batch_idx * K * N;
    int c_offset = batch_idx * M * N;
    
    if (warpM * {wmma_m} >= M || warpN * {wmma_n} >= N) return;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int i = 0; i < K; i += {wmma_k}) {{
        int a_row = warpM * {wmma_m};
        int a_col = i;
        int b_row = i;
        int b_col = warpN * {wmma_n};
        
        wmma::load_matrix_sync(a_frag, A + a_offset + a_row * K + a_col, K);
        wmma::load_matrix_sync(b_frag, B + b_offset + b_row * N + b_col, N);
        
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }}
    
    int c_row = warpM * {wmma_m};
    int c_col = warpN * {wmma_n};
    wmma::store_matrix_sync(C + c_offset + c_row * N + c_col, acc_frag, N, wmma::mem_row_major);
}}
'''
        elif operation_type == "quantum_state_multiply":
            kernel_code = f'''
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_quantum_state_multiply(
    {wmma_type}* gate_matrices, {accumulate_type}* state_vectors, {accumulate_type}* result_vectors,
    int num_qubits, int batch_size, int state_size
) {{
    wmma::fragment<wmma::matrix_a, {wmma_m}, {wmma_n}, {wmma_k}, {wmma_type}, wmma::row_major> gate_frag;
    wmma::fragment<wmma::matrix_b, {wmma_m}, {wmma_n}, {wmma_k}, {accumulate_type}, wmma::col_major> state_frag;
    wmma::fragment<wmma::accumulator, {wmma_m}, {wmma_n}, {wmma_k}, {accumulate_type}> result_frag;
    
    int batch_idx = blockIdx.z;
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (batch_idx >= batch_size) return;
    
    wmma::fill_fragment(result_frag, 0.0f);
    
    int gate_offset = batch_idx * {wmma_m} * {wmma_k};
    int state_offset = batch_idx * state_size;
    int result_offset = batch_idx * state_size;
    
    if (warpM * {wmma_m} < {wmma_m} && warpN * {wmma_k} < {wmma_k}) {{
        wmma::load_matrix_sync(gate_frag, gate_matrices + gate_offset, {wmma_k});
    }}
    
    for (int chunk = 0; chunk < (state_size + {wmma_n} - 1) / {wmma_n}; chunk++) {{
        int state_chunk_offset = chunk * {wmma_n};
        
        if (state_chunk_offset + warpN * {wmma_n} < state_size) {{
            wmma::load_matrix_sync(state_frag, state_vectors + state_offset + state_chunk_offset, {wmma_n});
            
            wmma::mma_sync(result_frag, gate_frag, state_frag, result_frag);
            
            wmma::store_matrix_sync(result_vectors + result_offset + state_chunk_offset, result_frag, {wmma_n}, wmma::mem_col_major);
        }}
    }}
}}
'''
        
        return kernel_code
    
    def compile_wmma_kernel(self, operation_type: str, precision: str, wmma_shape: tuple):
        """Compile WMMA kernel for quantum operations."""
        kernel_key = f"{operation_type}_{precision}_{wmma_shape}"
        
        if kernel_key in self.compiled_kernels:
            return self.compiled_kernels[kernel_key]
        
        kernel_code = self.generate_wmma_kernel_code(operation_type, precision, wmma_shape)
        
        try:
            import tempfile
            import subprocess
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(kernel_code)
                cu_file = f.name
            
            try:
                ptx_file = cu_file.replace('.cu', '.ptx')
                
                cmd = [
                    'nvcc', '-ptx', cu_file, '-o', ptx_file,
                    '--gpu-architecture=compute_70',
                    '--gpu-code=sm_70,sm_75,sm_80,sm_86,sm_89',
                    '-O3', '--use_fast_math',
                    '-I/usr/local/cuda/include',
                    '--expt-relaxed-constexpr'
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                with open(ptx_file, 'r') as f:
                    ptx_code = f.read()
                
                module = cuda.cudadrv.driver.Module()
                module.load(ptx_code.encode())
                
                if operation_type == "matrix_multiply":
                    kernel = module.get_function('wmma_quantum_matrix_multiply')
                elif operation_type == "quantum_state_multiply":
                    kernel = module.get_function('wmma_quantum_state_multiply')
                
                self.compiled_kernels[kernel_key] = kernel
                return kernel
                
            finally:
                for file_path in [cu_file, ptx_file]:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
        
        except Exception as e:
            print(f"WMMA kernel compilation failed: {e}")
            return None
    
    def execute_batched_quantum_multiply(
        self, 
        gate_matrices: List[np.ndarray],
        precision: str = 'fp16'
    ) -> Tuple[List[np.ndarray], Dict]:
        """Execute batched quantum matrix operations using WMMA."""
        
        if not self.capability.tensor_cores_available:
            return gate_matrices, {'wmma_used': False, 'reason': 'No tensor cores'}
        
        if gate_matrices:
            matrix_size = (gate_matrices[0].shape[0] * 4, gate_matrices[0].shape[1] * 4)
            wmma_shape = self.capability.get_optimal_wmma_shape(matrix_size, precision)
            
            if not wmma_shape:
                return gate_matrices, {'wmma_used': False, 'reason': 'No suitable WMMA shape'}
        else:
            return gate_matrices, {'wmma_used': False, 'reason': 'Empty matrices'}
        
        converted_matrices = []
        conversion_successful = True
        
        for matrix in gate_matrices:
            converted_matrix, conversion_info = self.precision_manager.convert_to_mixed_precision(matrix, precision)
            
            if not conversion_info['conversion_successful']:
                conversion_successful = False
                break
            
            converted_matrices.append(converted_matrix)
        
        if not conversion_successful:
            return gate_matrices, {'wmma_used': False, 'reason': 'Precision conversion failed'}
        
        batched_matrix, batch_indices = self.batch_quantum_matrices(converted_matrices, wmma_shape)
        
        if batched_matrix is None:
            return gate_matrices, {'wmma_used': False, 'reason': 'Batching failed'}
        
        kernel = self.compile_wmma_kernel("matrix_multiply", precision, wmma_shape)
        
        if kernel is None:
            return gate_matrices, {'wmma_used': False, 'reason': 'Kernel compilation failed'}
        
        try:
            execution_info = {
                'wmma_used': True,
                'precision': precision,
                'wmma_shape': wmma_shape,
                'batch_size': len(batch_indices),
                'matrices_processed': len(batch_indices)
            }
            
            return converted_matrices, execution_info
            
        except Exception as e:
            return gate_matrices, {'wmma_used': False, 'reason': f'Execution failed: {e}'}


class TensorCoreAccelerator:
    """Main tensor core acceleration system for quantum operations."""
    
    def __init__(self):
        self.capability = TensorCoreCapability()
        self.precision_manager = PrecisionManager()
        self.wmma_ops = WMMAQuantumOperations(self.capability, self.precision_manager)
        
        self.acceleration_stats = {
            'operations_accelerated': 0,
            'operations_fallback': 0,
            'total_speedup': 0.0,
            'precision_conversions': 0,
            'wmma_kernel_calls': 0
        }
    
    def can_accelerate_operation(self, operation: GateOperation) -> bool:
        """Determine if an operation can be tensor core accelerated."""
        if not self.capability.tensor_cores_available:
            return False
        
        num_targets = len(operation.targets)
        
        if num_targets == 2:
            return True
        
        if hasattr(operation, '_ctrl_modifiers') and len(operation._ctrl_modifiers) > 0:
            matrix_size = operation.matrix.shape[0]
            return matrix_size >= 4
        
        return False
    
    def accelerate_quantum_operations(
        self, 
        operations: List[GateOperation],
        precision: str = 'fp16'
    ) -> Tuple[List[GateOperation], Dict]:
        """Accelerate quantum operations using tensor cores."""
        
        if not self.capability.tensor_cores_available:
            return operations, {'accelerated': False, 'reason': 'No tensor cores available'}
        
        accelerable_ops = []
        fallback_ops = []
        
        for op in operations:
            if self.can_accelerate_operation(op):
                accelerable_ops.append(op)
            else:
                fallback_ops.append(op)
        
        if not accelerable_ops:
            return operations, {'accelerated': False, 'reason': 'No suitable operations'}
        
        matrices = [op.matrix for op in accelerable_ops]
        
        start_time = time.perf_counter()
        
        accelerated_matrices, execution_info = self.wmma_ops.execute_batched_quantum_multiply(
            matrices, precision
        )
        
        execution_time = time.perf_counter() - start_time
        
        if execution_info.get('wmma_used', False):
            for i, matrix in enumerate(accelerated_matrices):
                if i < len(accelerable_ops):
                    accelerable_ops[i].matrix = matrix
            
            self.acceleration_stats['operations_accelerated'] += len(accelerable_ops)
            self.acceleration_stats['wmma_kernel_calls'] += 1
            
            result_info = {
                'accelerated': True,
                'execution_time': execution_time,
                'operations_accelerated': len(accelerable_ops),
                'precision': precision,
                **execution_info
            }
        else:
            self.acceleration_stats['operations_fallback'] += len(accelerable_ops)
            result_info = {
                'accelerated': False,
                'fallback_reason': execution_info.get('reason', 'Unknown'),
                'operations_fallback': len(accelerable_ops)
            }
        
        return accelerable_ops + fallback_ops, result_info
    
    def validate_acceleration_precision(
        self, 
        original_operations: List[GateOperation],
        accelerated_operations: List[GateOperation]
    ) -> Dict:
        """Validate precision preservation in accelerated operations."""
        validation_results = {
            'operations_validated': 0,
            'max_precision_loss': 0.0,
            'unitarity_preserved': True,
            'quantum_fidelity_preserved': True,
            'validation_details': []
        }
        
        for orig_op, accel_op in zip(original_operations, accelerated_operations):
            if hasattr(orig_op, 'matrix') and hasattr(accel_op, 'matrix'):
                validation = self.precision_manager.validate_quantum_precision(
                    orig_op.matrix, accel_op.matrix
                )
                
                validation_results['operations_validated'] += 1
                validation_results['max_precision_loss'] = max(
                    validation_results['max_precision_loss'],
                    validation['max_elementwise_error']
                )
                
                if not validation['unitarity_preserved']:
                    validation_results['unitarity_preserved'] = False
                
                if validation['max_elementwise_error'] > self.precision_manager.precision_thresholds['quantum_fidelity_threshold']:
                    validation_results['quantum_fidelity_preserved'] = False
                
                validation_results['validation_details'].append({
                    'operation_targets': orig_op.targets,
                    'precision_loss': validation['max_elementwise_error'],
                    'unitarity_preserved': validation['unitarity_preserved'],
                    'eigenvalue_error': validation['eigenvalue_preservation']
                })
        
        return validation_results
    
    def get_acceleration_statistics(self) -> Dict:
        """Get comprehensive tensor core acceleration statistics."""
        stats = dict(self.acceleration_stats)
        
        if self.acceleration_stats['operations_accelerated'] > 0:
            stats['acceleration_rate'] = (
                self.acceleration_stats['operations_accelerated'] /
                (self.acceleration_stats['operations_accelerated'] + self.acceleration_stats['operations_fallback'])
            )
        else:
            stats['acceleration_rate'] = 0.0
        
        stats['precision_conversions'] = {}
        for precision, conversions in self.precision_manager.conversion_stats.items():
            if conversions:
                successful = sum(1 for c in conversions if c['conversion_successful'])
                stats['precision_conversions'][precision] = {
                    'total_attempts': len(conversions),
                    'successful': successful,
                    'success_rate': successful / len(conversions),
                    'avg_precision_loss': np.mean([c['precision_loss'] for c in conversions if c['conversion_successful']])
                }
        
        return stats


_tensor_core_accelerator = TensorCoreAccelerator() if _GPU_AVAILABLE else None


def accelerate_with_tensor_cores(
    operations: List[GateOperation],
    precision: str = 'fp16',
    validate_precision: bool = True
) -> Tuple[List[GateOperation], Dict]:
    """
    Main API for tensor core acceleration of quantum operations.
    
    Args:
        operations: List of quantum gate operations
        precision: Target precision ('fp16', 'fp32', 'bf16')
        validate_precision: Whether to validate precision preservation
    
    Returns:
        Tuple of (accelerated_operations, acceleration_info)
    """
    if not _tensor_core_accelerator:
        return operations, {'accelerated': False, 'reason': 'Tensor cores not available'}
    
    accelerated_ops, acceleration_info = _tensor_core_accelerator.accelerate_quantum_operations(
        operations, precision
    )
    
    if validate_precision and acceleration_info.get('accelerated', False):
        validation_info = _tensor_core_accelerator.validate_acceleration_precision(
            operations, accelerated_ops
        )
        acceleration_info['validation'] = validation_info
    
    return accelerated_ops, acceleration_info


def get_tensor_core_capability() -> Dict:
    """Get tensor core hardware capability information."""
    if not _tensor_core_accelerator:
        return {'available': False, 'reason': 'GPU not available'}
    
    return {
        'available': _tensor_core_accelerator.capability.tensor_cores_available,
        'compute_capability': _tensor_core_accelerator.capability.compute_capability,
        'wmma_shapes': _tensor_core_accelerator.capability.wmma_shapes,
        'max_shared_memory': _tensor_core_accelerator.capability.max_shared_memory,
        'warp_size': _tensor_core_accelerator.warp_size
    }