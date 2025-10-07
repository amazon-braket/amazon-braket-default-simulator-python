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
Advanced JIT circuit compiler for fused GPU kernel generation.

This module provides sophisticated circuit fusion capabilities that can compile
quantum circuits into optimized CUDA kernels for high-performance execution.
"""

from typing import Dict, List, Optional, Tuple
import hashlib
import os
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
from numba import cuda

from braket.default_simulator.operation import GateOperation
from braket.default_simulator.linalg_utils import (
    DIAGONAL_GATES,
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
    _OPTIMAL_THREADS_PER_BLOCK,
)

from braket.default_simulator.tensor_core_acceleration import (
    _tensor_core_accelerator,
    accelerate_with_tensor_cores,
    get_tensor_core_capability,
)

_TENSOR_CORES_AVAILABLE = _tensor_core_accelerator is not None


class CircuitPattern:
    """Represents a fusible circuit pattern with dependency analysis."""
    
    def __init__(self, operations: List[GateOperation]):
        self.operations = operations
        self.qubit_dependencies = self._analyze_dependencies()
        self.pattern_signature = self._compute_pattern_signature()
        self.fusion_segments = self._identify_fusion_segments()
    
    def _analyze_dependencies(self) -> Dict[int, List[int]]:
        """Analyze qubit dependencies between operations."""
        deps = defaultdict(list)
        qubit_last_used = {}
        
        for i, op in enumerate(self.operations):
            op_qubits = set(op.targets) | set(getattr(op, "_ctrl_modifiers", []))
            
            for qubit in op_qubits:
                if qubit in qubit_last_used:
                    deps[i].append(qubit_last_used[qubit])
                qubit_last_used[qubit] = i
        
        return deps
    
    def _compute_pattern_signature(self) -> str:
        """Compute signature based on gate sequence and qubit pattern."""
        gate_sequence = []
        qubit_pattern = []
        
        for op in self.operations:
            gate_type = getattr(op, "gate_type", "custom")
            gate_sequence.append(gate_type)
            qubit_pattern.extend(op.targets)
        
        pattern_str = f"gates:{','.join(gate_sequence)};qubits:{','.join(map(str, qubit_pattern))}"
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
    
    def _identify_fusion_segments(self) -> List[Tuple[int, int]]:
        """Identify segments of operations that can be fused together."""
        segments = []
        segment_start = 0
        
        for i in range(1, len(self.operations)):
            if not self._can_fuse_with_segment(i, segment_start):
                segments.append((segment_start, i))
                segment_start = i
        
        if segment_start < len(self.operations):
            segments.append((segment_start, len(self.operations)))
        
        return segments
    
    def _can_fuse_with_segment(self, op_idx: int, segment_start: int) -> bool:
        """Check if operation can be fused with current segment."""
        if op_idx - segment_start >= 20:
            return False
        
        current_qubits = set(self.operations[op_idx].targets)
        
        overlapping_ops = 0
        for j in range(segment_start, op_idx):
            prev_qubits = set(self.operations[j].targets)
            if current_qubits & prev_qubits:
                overlapping_ops += 1
                if overlapping_ops > 2:
                    return False
        
        return True


class KernelTemplateLibrary:
    """Library of optimized CUDA kernel templates for common gate patterns."""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize common kernel templates."""
        self.templates["single_chain"] = self._single_qubit_chain_template()
        self.templates["mixed_gates"] = self._mixed_gates_template()
        self.templates["diagonal_chain"] = self._diagonal_chain_template()
    
    def _single_qubit_chain_template(self) -> str:
        """Template for chains of single-qubit gates."""
        return """
extern "C" __global__ void single_chain_kernel(
    cuDoubleComplex* state_flat,
    cuDoubleComplex* out_flat,
    cuDoubleComplex* gate_matrices,
    int* gate_targets,
    int num_gates,
    int total_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < total_size) {
        cuDoubleComplex amplitude = state_flat[i];
        
        for (int g = 0; g < num_gates; g++) {
            int target = gate_targets[g];
            int target_bit = target;
            int paired_idx = i ^ (1 << target_bit);
            
            if (i <= paired_idx) {
                cuDoubleComplex s0 = (i < paired_idx) ? amplitude : state_flat[paired_idx];
                cuDoubleComplex s1 = (i < paired_idx) ? state_flat[paired_idx] : amplitude;
                
                cuDoubleComplex* matrix = &gate_matrices[g * 4];
                cuDoubleComplex new_s0 = cuCadd(cuCmul(matrix[0], s0), cuCmul(matrix[1], s1));
                cuDoubleComplex new_s1 = cuCadd(cuCmul(matrix[2], s0), cuCmul(matrix[3], s1));
                
                amplitude = (i < paired_idx) ? new_s0 : new_s1;
                if (i != paired_idx) {
                    cuDoubleComplex other_amp = (i < paired_idx) ? new_s1 : new_s0;
                    atomicAdd(&out_flat[paired_idx].x, other_amp.x - state_flat[paired_idx].x);
                    atomicAdd(&out_flat[paired_idx].y, other_amp.y - state_flat[paired_idx].y);
                }
            }
        }
        
        out_flat[i] = amplitude;
        i += stride;
    }
}
"""
    
    def _mixed_gates_template(self) -> str:
        """Template for mixed single-qubit and two-qubit gates."""
        return """
extern "C" __global__ void mixed_gates_kernel(
    cuDoubleComplex* state_flat,
    cuDoubleComplex* out_flat,
    int* gate_types,
    int* gate_params,
    cuDoubleComplex* gate_data,
    int num_gates,
    int total_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < total_size) {
        cuDoubleComplex amplitude = state_flat[i];
        
        for (int g = 0; g < num_gates; g++) {
            int gate_type = gate_types[g];
            
            if (gate_type == 1) {
                int target = gate_params[g * 4];
                int target_bit = target;
                
                int paired_idx = i ^ (1 << target_bit);
                if (i <= paired_idx) {
                    cuDoubleComplex s0 = amplitude;
                    cuDoubleComplex s1 = (i == paired_idx) ? amplitude : state_flat[paired_idx];
                    
                    cuDoubleComplex* matrix = &gate_data[g * 4];
                    amplitude = cuCadd(cuCmul(matrix[0], s0), cuCmul(matrix[1], s1));
                    
                    if (i != paired_idx) {
                        cuDoubleComplex new_s1 = cuCadd(cuCmul(matrix[2], s0), cuCmul(matrix[3], s1));
                        atomicAdd(&out_flat[paired_idx].x, new_s1.x - state_flat[paired_idx].x);
                        atomicAdd(&out_flat[paired_idx].y, new_s1.y - state_flat[paired_idx].y);
                    }
                }
            }
            else if (gate_type == 2) {
                int control = gate_params[g * 4];
                int target = gate_params[g * 4 + 1];
                int control_bit = control;
                int target_bit = target;
                
                if ((i >> control_bit) & 1) {
                    int swap_idx = i ^ (1 << target_bit);
                    cuDoubleComplex temp = amplitude;
                    amplitude = state_flat[swap_idx];
                    atomicExch(&out_flat[swap_idx].x, temp.x);
                    atomicExch(&out_flat[swap_idx].y, temp.y);
                }
            }
        }
        
        out_flat[i] = amplitude;
        i += stride;
    }
}
"""
    
    def _diagonal_chain_template(self) -> str:
        """Optimized template for diagonal gate chains."""
        return """
extern "C" __global__ void diagonal_chain_kernel(
    cuDoubleComplex* state_flat,
    cuDoubleComplex* out_flat,
    cuDoubleComplex* phase_factors,
    int* gate_targets,
    int num_gates,
    int total_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < total_size) {
        cuDoubleComplex amplitude = state_flat[i];
        
        for (int g = 0; g < num_gates; g++) {
            int target = gate_targets[g];
            int target_bit = target;
            
            if ((i >> target_bit) & 1) {
                amplitude = cuCmul(amplitude, phase_factors[g * 2 + 1]);
            } else {
                amplitude = cuCmul(amplitude, phase_factors[g * 2]);
            }
        }
        
        out_flat[i] = amplitude;
        i += stride;
    }
}
"""
    
    def get_template(self, pattern_type: str) -> str | None:
        """Get kernel template for given pattern type."""
        return self.templates.get(pattern_type)


class CircuitCompiler:
    """Advanced JIT circuit compiler for fused GPU kernel generation."""
    
    def __init__(self):
        self.compiled_kernels: dict[str, cuda.cudadrv.driver.Function] = {}
        self.fusion_cache: dict[str, str | None] = {}
        self.template_library = KernelTemplateLibrary()
        self.pattern_cache: dict[str, CircuitPattern] = {}
        
        self.supported_gates = {
            "pauli_x", "pauli_y", "pauli_z", "h", "s", "si", "t", "ti",
            "rx", "ry", "rz", "phaseshift", "cx", "cz", "swap", "cphaseshift",
            None
        }
    
    def can_fuse_circuit(self, operations: List[GateOperation], max_gates: int = 20) -> bool:
        """Advanced circuit fusion analysis with tensor core consideration."""
        if len(operations) == 0 or len(operations) > max_gates:
            return False
        
        if not self._quick_feasibility_check(operations):
            return False
        
        if _TENSOR_CORES_AVAILABLE and self._can_use_tensor_cores(operations):
            return True
        
        pattern = self._get_or_create_pattern(operations)
        return len(pattern.fusion_segments) > 0 and self._has_beneficial_fusion(pattern)
    
    def _can_use_tensor_cores(self, operations: List[GateOperation]) -> bool:
        """Check if operations are suitable for tensor core acceleration."""
        if not _TENSOR_CORES_AVAILABLE:
            return False
        
        tensor_suitable = 0
        for op in operations:
            if _tensor_core_accelerator.can_accelerate_operation(op):
                tensor_suitable += 1
        
        return tensor_suitable >= len(operations) * 0.5
    
    def try_tensor_core_acceleration(
        self, 
        operations: List[GateOperation], 
        precision: str = 'fp16'
    ) -> Tuple[List[GateOperation], Dict]:
        """Attempt tensor core acceleration for quantum operations."""
        if not _TENSOR_CORES_AVAILABLE:
            return operations, {'accelerated': False, 'reason': 'Tensor cores not available'}
        
        return accelerate_with_tensor_cores(operations, precision, validate_precision=True)
    
    def _quick_feasibility_check(self, operations: List[GateOperation]) -> bool:
        """Quick check for obvious fusion blockers."""
        for op in operations:
            gate_type = getattr(op, "gate_type", None)
            num_targets = len(op.targets)
            num_controls = len(getattr(op, "_ctrl_modifiers", []))
            
            if num_targets > 2 or num_controls > 2:
                return False
            
            if gate_type not in self.supported_gates:
                if num_targets > 2:
                    return False
        
        return True
    
    def _get_or_create_pattern(self, operations: list[GateOperation]) -> CircuitPattern:
        """Get or create circuit pattern analysis."""
        op_signature = "|".join([
            f"{getattr(op, 'gate_type', 'custom')}:{','.join(map(str, op.targets))}"
            for op in operations
        ])
        signature_hash = hashlib.md5(op_signature.encode()).hexdigest()
        
        if signature_hash not in self.pattern_cache:
            self.pattern_cache[signature_hash] = CircuitPattern(operations)
        
        return self.pattern_cache[signature_hash]
    
    def _has_beneficial_fusion(self, pattern: CircuitPattern) -> bool:
        """Determine if fusion provides performance benefit."""
        max_segment_length = max(
            segment[1] - segment[0] for segment in pattern.fusion_segments
        ) if pattern.fusion_segments else 0
        
        return max_segment_length >= 3
    
    def generate_circuit_signature(self, operations: list[GateOperation]) -> str:
        """Generate unique signature for circuit pattern."""
        signature_parts = []
        for op in operations:
            gate_type = getattr(op, "gate_type", "custom")
            targets = ",".join(map(str, op.targets))
            controls = ",".join(map(str, getattr(op, "_ctrl_modifiers", [])))
            matrix_hash = hash(op.matrix.tobytes()) if hasattr(op, 'matrix') else 0
            signature_parts.append(f"{gate_type}:t{targets}:c{controls}:m{matrix_hash}")
        
        full_signature = "|".join(signature_parts)
        return hashlib.md5(full_signature.encode()).hexdigest()
    
    def compile_fused_circuit(
        self, 
        operations: list[GateOperation], 
        qubit_count: int
    ) -> Optional[cuda.cudadrv.driver.Function]:
        """Compile circuit into fused CUDA kernel."""
        circuit_signature = self.generate_circuit_signature(operations)
        
        if circuit_signature in self.compiled_kernels:
            return self.compiled_kernels[circuit_signature]
        
        if circuit_signature in self.fusion_cache:
            return self.fusion_cache[circuit_signature]
        
        if not self.can_fuse_circuit(operations):
            self.fusion_cache[circuit_signature] = None
            return None
        
        fused_kernel = self._generate_fused_kernel(operations, qubit_count)
        if fused_kernel:
            self.compiled_kernels[circuit_signature] = fused_kernel
            return fused_kernel
        else:
            self.fusion_cache[circuit_signature] = None
            return None
    
    def _generate_fused_kernel(
        self, 
        operations: list[GateOperation], 
        qubit_count: int
    ) -> Optional[cuda.cudadrv.driver.Function]:
        """Generate optimized fused CUDA kernel for circuit."""
        kernel_code = self._build_kernel_code(operations, qubit_count)
        if not kernel_code:
            return None
        
        compiled_kernel = self._compile_cuda_code(kernel_code)
        return compiled_kernel
    
    def _build_kernel_code(self, operations: list[GateOperation], qubit_count: int) -> str | None:
        """Build CUDA kernel code for fused circuit execution."""
        if len(operations) > 20:
            return None
        
        operation_code = []
        temp_vars = []
        
        for i, op in enumerate(operations):
            gate_type = getattr(op, "gate_type", None)
            targets = op.targets
            matrix = op.matrix
            
            if len(targets) == 1 and not getattr(op, "_ctrl_modifiers", []):
                code, temps = self._generate_single_qubit_code(
                    gate_type, targets[0], matrix, i, qubit_count
                )
                operation_code.append(code)
                temp_vars.extend(temps)
            elif len(targets) == 2 and not getattr(op, "_ctrl_modifiers", []):
                if gate_type == "cx":
                    code, temps = self._generate_cnot_code(targets[0], targets[1], i, qubit_count)
                    operation_code.append(code)
                    temp_vars.extend(temps)
                else:
                    return None
            else:
                return None
        
        if not operation_code:
            return None
        
        return self._assemble_kernel_code(operation_code, temp_vars, qubit_count)
    
    def _generate_single_qubit_code(
        self, gate_type: str, target: int, matrix: np.ndarray, op_index: int, qubit_count: int
    ) -> Tuple[str, list[str]]:
        """Generate CUDA code for single-qubit gate."""
        target_bit = qubit_count - target - 1
        
        if gate_type in DIAGONAL_GATES:
            a, d = matrix[0, 0], matrix[1, 1]
            code = f"""
    if (i & (1 << {target_bit})) {{
        temp_amp *= {d.real} + {d.imag}j;
    }} else {{
        temp_amp *= {a.real} + {a.imag}j;
    }}
"""
            return code, []
        else:
            a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
            temp_var = f"paired_idx_{op_index}"
            code = f"""
    {temp_var} = i ^ (1 << {target_bit});
    if (i <= {temp_var}) {{
        complex128 s0 = temp_amp;
        complex128 s1 = (i == {temp_var}) ? temp_amp : state_flat[{temp_var}];
        temp_amp = ({a.real} + {a.imag}j) * s0 + ({b.real} + {b.imag}j) * s1;
        if (i != {temp_var}) {{
            atomicAdd(&out_flat[{temp_var}].x, 
                      (({c.real} + {c.imag}j) * s0 + ({d.real} + {d.imag}j) * s1).x - state_flat[{temp_var}].x);
            atomicAdd(&out_flat[{temp_var}].y, 
                      (({c.real} + {c.imag}j) * s0 + ({d.real} + {d.imag}j) * s1).y - state_flat[{temp_var}].y);
        }}
    }}
"""
            return code, [temp_var]
    
    def _generate_cnot_code(
        self, control: int, target: int, op_index: int, qubit_count: int
    ) -> Tuple[str, list[str]]:
        """Generate CUDA code for CNOT gate."""
        control_bit = qubit_count - control - 1
        target_bit = qubit_count - target - 1
        
        code = f"""
    if ((i >> {control_bit}) & 1) {{
        int swap_idx = i ^ (1 << {target_bit});
        complex128 temp = temp_amp;
        temp_amp = state_flat[swap_idx];
        atomicExch(&out_flat[swap_idx].x, temp.x);
        atomicExch(&out_flat[swap_idx].y, temp.y);
    }}
"""
        return code, []
    
    def _assemble_kernel_code(
        self, operation_code: list[str], temp_vars: list[str], qubit_count: int
    ) -> str:
        """Assemble complete CUDA kernel code."""
        temp_declarations = "\n".join([f"    int {var};" for var in temp_vars])
        operations = "\n".join(operation_code)
        
        kernel_code = f"""
extern "C" __global__ void fused_circuit_kernel(
    complex128* state_flat,
    complex128* out_flat,
    int total_size
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < total_size) {{
{temp_declarations}
        complex128 temp_amp = state_flat[i];
        
{operations}
        
        out_flat[i] = temp_amp;
        i += stride;
    }}
}}
"""
        return kernel_code
    
    def _compile_cuda_code(self, kernel_code: str) -> cuda.cudadrv.driver.Function:
        """Compile CUDA C code into executable kernel function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            cu_file = f.name
        
        ptx_file = cu_file.replace('.cu', '.ptx')
        
        cmd = [
            'nvcc', '-ptx', cu_file, '-o', ptx_file,
            '--gpu-architecture=compute_75',
            '--gpu-code=sm_75',
            '-O3', '--use_fast_math'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        with open(ptx_file, 'r') as f:
            ptx_code = f.read()
        
        module = cuda.cudadrv.driver.Module()
        module.load(ptx_code.encode())
        kernel = module.get_function('fused_circuit_kernel')
        
        for file_path in [cu_file, ptx_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        return kernel
    
    def execute_fused_kernel(
        self,
        kernel: cuda.cudadrv.driver.Function,
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray
    ):
        """Execute compiled fused kernel."""
        total_size = state_gpu.size
        
        threads_per_block = 512
        blocks_per_grid = max(
            min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID),
            256
        )
        
        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)
        
        kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, total_size
        )
    
    def analyze_pattern_type(self, operations: List[GateOperation]) -> str:
        """Analyze circuit pattern to determine optimal kernel template."""
        single_qubit_count = 0
        two_qubit_count = 0
        diagonal_count = 0
        
        for op in operations:
            gate_type = getattr(op, "gate_type", None)
            num_targets = len(op.targets)
            
            if num_targets == 1:
                single_qubit_count += 1
                if gate_type in DIAGONAL_GATES:
                    diagonal_count += 1
            elif num_targets == 2:
                two_qubit_count += 1
        
        if diagonal_count == len(operations):
            return "diagonal_chain"
        elif single_qubit_count > 0 and two_qubit_count > 0:
            return "mixed_gates"
        elif single_qubit_count == len(operations):
            return "single_chain"
        else:
            return "custom"
    
    def compile_template_kernel(
        self, 
        operations: List[GateOperation], 
        pattern_type: str,
        qubit_count: int
    ) -> Optional[cuda.cudadrv.driver.Function]:
        """Compile circuit using optimized template kernel."""
        template_code = self.template_library.get_template(pattern_type)
        if not template_code:
            return None
        
        return self._compile_cuda_code(template_code)
    
    def execute_template_kernel(
        self,
        kernel: cuda.cudadrv.driver.Function,
        operations: list[GateOperation],
        state_gpu: cuda.devicearray.DeviceNDArray,
        out_gpu: cuda.devicearray.DeviceNDArray,
        qubit_count: int,
        pattern_type: str
    ):
        """Execute template kernel with operation-specific data."""
        total_size = state_gpu.size
        
        threads_per_block = 512
        blocks_per_grid = max(
            min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID),
            256
        )
        
        if pattern_type == "single_chain":
            self._execute_single_chain_kernel(
                kernel, operations, state_gpu, out_gpu, 
                threads_per_block, blocks_per_grid, total_size
            )
        elif pattern_type == "mixed_gates":
            self._execute_mixed_gates_kernel(
                kernel, operations, state_gpu, out_gpu,
                threads_per_block, blocks_per_grid, total_size
            )
        elif pattern_type == "diagonal_chain":
            self._execute_diagonal_chain_kernel(
                kernel, operations, state_gpu, out_gpu,
                threads_per_block, blocks_per_grid, total_size
            )
    
    def _execute_single_chain_kernel(
        self, kernel, operations, state_gpu, out_gpu,
        threads_per_block, blocks_per_grid, total_size
    ):
        """Execute single-qubit chain template kernel."""
        gate_matrices = []
        gate_targets = []
        
        for op in operations:
            if len(op.targets) == 1:
                gate_matrices.extend(op.matrix.flatten())
                gate_targets.append(op.targets[0])
        
        matrices_gpu = cuda.to_device(np.array(gate_matrices, dtype=np.complex128))
        targets_gpu = cuda.to_device(np.array(gate_targets, dtype=np.int32))
        
        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)
        
        kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, matrices_gpu, targets_gpu, 
            len(operations), total_size
        )
    
    def _execute_mixed_gates_kernel(
        self, kernel, operations, state_gpu, out_gpu,
        threads_per_block, blocks_per_grid, total_size
    ):
        """Execute mixed gates template kernel."""
        gate_types = []
        gate_params = []
        gate_data = []
        
        for op in operations:
            num_targets = len(op.targets)
            
            if num_targets == 1:
                gate_types.append(1)
                gate_params.extend([op.targets[0], 0, 0, 0])
                gate_data.extend(op.matrix.flatten())
            elif num_targets == 2 and getattr(op, "gate_type", None) == "cx":
                gate_types.append(2)
                gate_params.extend([op.targets[0], op.targets[1], 0, 0])
                gate_data.extend([0, 0, 0, 0])
        
        types_gpu = cuda.to_device(np.array(gate_types, dtype=np.int32))
        params_gpu = cuda.to_device(np.array(gate_params, dtype=np.int32))
        data_gpu = cuda.to_device(np.array(gate_data, dtype=np.complex128))
        
        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)
        
        kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, types_gpu, params_gpu, data_gpu,
            len(operations), total_size
        )
    
    def _execute_diagonal_chain_kernel(
        self, kernel, operations, state_gpu, out_gpu,
        threads_per_block, blocks_per_grid, total_size
    ):
        """Execute diagonal gates chain template kernel."""
        phase_factors = []
        gate_targets = []
        
        for op in operations:
            if len(op.targets) == 1:
                a, d = op.matrix[0, 0], op.matrix[1, 1]
                phase_factors.extend([a, d])
                gate_targets.append(op.targets[0])
        
        factors_gpu = cuda.to_device(np.array(phase_factors, dtype=np.complex128))
        targets_gpu = cuda.to_device(np.array(gate_targets, dtype=np.int32))
        
        state_flat = state_gpu.reshape(-1)
        out_flat = out_gpu.reshape(-1)
        
        kernel[blocks_per_grid, threads_per_block](
            state_flat, out_flat, factors_gpu, targets_gpu,
            len(operations), total_size
        )
    
    def clear_cache(self):
        """Clear compiled kernel cache."""
        self.compiled_kernels.clear()
        self.fusion_cache.clear()
        self.pattern_cache.clear()


_circuit_compiler = CircuitCompiler()


def compile_and_execute_circuit(
    operations: list[GateOperation],
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    qubit_count: int
) -> bool:
    """Compile and execute circuit as fused kernel if possible."""
    if len(operations) < 3 or len(operations) > 20:
        return False
    
    fused_kernel = _circuit_compiler.compile_fused_circuit(operations, qubit_count)
    if fused_kernel:
        _circuit_compiler.execute_fused_kernel(fused_kernel, state_gpu, out_gpu)
        return True
    
    return False


@cuda.jit(inline=True, fastmath=True)
def _fused_gate_sequence_kernel(state_flat, out_flat, gate_data, num_gates, n_qubits, total_size):
    """Simplified template kernel for reliable gate fusion."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        out_flat[i] = state_flat[i]
        
        for gate_idx in range(num_gates):
            gate_type = int(gate_data[gate_idx, 0].real)
            target = int(gate_data[gate_idx, 1].real)
            
            if gate_type == 1:
                target_bit = n_qubits - target - 1
                paired_idx = i ^ (1 << target_bit)
                if i == paired_idx or i < paired_idx:
                    temp = out_flat[i]
                    out_flat[i] = out_flat[paired_idx] if i != paired_idx else temp
                    if i != paired_idx:
                        out_flat[paired_idx] = temp
            
            elif gate_type == 3:
                target_bit = n_qubits - target - 1
                if (i >> target_bit) & 1:
                    out_flat[i] *= -1
            
            elif gate_type == 4:
                target_bit = n_qubits - target - 1
                paired_idx = i ^ (1 << target_bit)
                if i <= paired_idx:
                    s0 = out_flat[i]
                    s1 = out_flat[paired_idx] if i != paired_idx else s0
                    inv_sqrt2 = 0.7071067811865476
                    out_flat[i] = inv_sqrt2 * (s0 + s1)
                    if i != paired_idx:
                        out_flat[paired_idx] = inv_sqrt2 * (s0 - s1)
            
            elif gate_type == 5:
                control = int(gate_data[gate_idx, 2].real)
                control_bit = n_qubits - control - 1
                target_bit = n_qubits - target - 1
                
                if (i >> control_bit) & 1:
                    swap_idx = i ^ (1 << target_bit)
                    temp = out_flat[i]
                    out_flat[i] = out_flat[swap_idx]
                    out_flat[swap_idx] = temp
        
        i += stride


def create_optimized_gate_sequence(operations: list[GateOperation], qubit_count: int) -> Optional[cuda.devicearray.DeviceNDArray]:
    """Create optimized gate sequence data for template kernel."""
    if len(operations) > 20:
        return None
    
    gate_data = np.zeros((len(operations), 8), dtype=np.complex128)
    
    for i, op in enumerate(operations):
        gate_type = getattr(op, "gate_type", None)
        targets = op.targets
        
        if gate_type == "pauli_x":
            gate_data[i, 0] = 1
            gate_data[i, 1] = targets[0]
        elif gate_type == "pauli_y":
            gate_data[i, 0] = 2  
            gate_data[i, 1] = targets[0]
        elif gate_type == "pauli_z":
            gate_data[i, 0] = 3
            gate_data[i, 1] = targets[0]
        elif gate_type == "h":
            gate_data[i, 0] = 4
            gate_data[i, 1] = targets[0]
        elif gate_type == "cx" and len(targets) == 2:
            gate_data[i, 0] = 5
            gate_data[i, 1] = targets[1]
            gate_data[i, 2] = targets[0]
        else:
            return None
    
    return cuda.to_device(gate_data)


def execute_template_fused_kernel(
    operations: list[GateOperation],
    state_gpu: cuda.devicearray.DeviceNDArray,
    out_gpu: cuda.devicearray.DeviceNDArray,
    qubit_count: int
) -> bool:
    """Execute operations using template fused kernel."""
    gate_data_gpu = create_optimized_gate_sequence(operations, qubit_count)
    if gate_data_gpu is None:
        return False
    
    total_size = state_gpu.size
    state_flat = state_gpu.reshape(-1)
    out_flat = out_gpu.reshape(-1)
    
    threads_per_block = 512
    blocks_per_grid = max(
        min((total_size + threads_per_block - 1) // threads_per_block, _MAX_BLOCKS_PER_GRID),
        256
    )
    
    _fused_gate_sequence_kernel[blocks_per_grid, threads_per_block](
        state_flat, out_flat, gate_data_gpu, len(operations), qubit_count, total_size
    )
    
    return True
