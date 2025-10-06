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
from typing import Tuple, Optional
import hashlib

from braket.default_simulator.operation import GateOperation
from braket.default_simulator.linalg_utils import (
    _OPTIMAL_THREADS_PER_BLOCK,
    _MAX_BLOCKS_PER_GRID,
    DIAGONAL_GATES,
)


class CircuitCompiler:
    """JIT circuit compiler for fused GPU kernel generation."""
    
    def __init__(self):
        self.compiled_kernels: dict[str, cuda.cudadrv.driver.Function] = {}
        self.fusion_cache: dict[str, Optional[str]] = {}
        
    def can_fuse_circuit(self, operations: list[GateOperation], max_gates: int = 20) -> bool:
        """Determine if circuit can be fused into single kernel."""
        if len(operations) == 0 or len(operations) > max_gates:
            return False
        
        qubit_usage = set()
        for op in operations:
            gate_type = getattr(op, "gate_type", None)
            num_targets = len(op.targets)
            num_controls = len(getattr(op, "_ctrl_modifiers", []))
            
            if num_targets > 2 or num_controls > 2:
                return False
            
            if gate_type not in self._get_supported_gates():
                return False
            
            op_qubits = set(op.targets)
            if op_qubits & qubit_usage:
                if not self._can_parallelize_with_previous(op, operations[:operations.index(op)]):
                    return False
            
            qubit_usage.update(op_qubits)
        
        return True
    
    def _get_supported_gates(self) -> set[str]:
        """Get set of gates supported by fusion compiler."""
        return {
            "pauli_x", "pauli_y", "pauli_z", "h", "s", "t", "rz", "ry", "rx",
            "cx", "swap", "cz", "cphaseshift", None  # None for custom matrices
        }
    
    def _can_parallelize_with_previous(self, op: GateOperation, prev_ops: list[GateOperation]) -> bool:
        """Check if operation can be parallelized with previous operations."""
        op_qubits = set(op.targets)
        for prev_op in reversed(prev_ops[-5:]):  # Check last 5 operations
            prev_qubits = set(prev_op.targets)
            if not (op_qubits & prev_qubits):
                return True
        return False
    
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
        
        try:
            fused_kernel = self._generate_fused_kernel(operations, qubit_count)
            if fused_kernel:
                self.compiled_kernels[circuit_signature] = fused_kernel
                return fused_kernel
            else:
                self.fusion_cache[circuit_signature] = None
                return None
        except Exception:
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
        
        try:
            compiled_kernel = self._compile_cuda_code(kernel_code)
            return compiled_kernel
        except Exception:
            return None
    
    def _build_kernel_code(self, operations: list[GateOperation], qubit_count: int) -> Optional[str]:
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
                    return None  # Complex two-qubit gates not supported yet
            else:
                return None  # Controlled gates not supported in fusion yet
        
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
    // Operation {op_index}: Diagonal gate on qubit {target}
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
    // Operation {op_index}: Single-qubit gate on qubit {target}
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
    // Operation {op_index}: CNOT gate control={control}, target={target}
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
            
            return kernel
        
        finally:
            for file_path in [cu_file, ptx_file]:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
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
    
    def clear_cache(self):
        """Clear compiled kernel cache."""
        self.compiled_kernels.clear()
        self.fusion_cache.clear()


# Global circuit compiler instance
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
        try:
            _circuit_compiler.execute_fused_kernel(fused_kernel, state_gpu, out_gpu)
            return True
        except Exception:
            return False
    
    return False


@cuda.jit(inline=True, fastmath=True)
def _fused_gate_sequence_kernel(state_flat, out_flat, gate_data, num_gates, n_qubits, total_size):
    """Template kernel for fused gate sequences."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while i < total_size:
        amplitude = state_flat[i]
        
        for gate_idx in range(num_gates):
            gate_type = int(gate_data[gate_idx, 0].real)
            target = int(gate_data[gate_idx, 1].real)
            
            if gate_type == 1:  # Pauli-X
                target_bit = n_qubits - target - 1
                paired_idx = i ^ (1 << target_bit)
                if i <= paired_idx and i != paired_idx:
                    temp = amplitude
                    amplitude = state_flat[paired_idx]
                    if i < paired_idx:
                        cuda.atomic.add(out_flat, paired_idx, temp - state_flat[paired_idx])
            
            elif gate_type == 2:  # Pauli-Y  
                target_bit = n_qubits - target - 1
                if (i >> target_bit) & 1:
                    amplitude *= -1j
                else:
                    amplitude *= 1j
                    paired_idx = i ^ (1 << target_bit)
                    if i != paired_idx:
                        cuda.atomic.add(out_flat, paired_idx, 1j * state_flat[i] - state_flat[paired_idx])
            
            elif gate_type == 3:  # Pauli-Z
                target_bit = n_qubits - target - 1
                if (i >> target_bit) & 1:
                    amplitude *= -1
            
            elif gate_type == 4:  # Hadamard
                target_bit = n_qubits - target - 1
                paired_idx = i ^ (1 << target_bit)
                if i <= paired_idx:
                    s0 = amplitude
                    s1 = state_flat[paired_idx] if i != paired_idx else amplitude
                    inv_sqrt2 = 0.7071067811865476
                    amplitude = inv_sqrt2 * (s0 + s1)
                    if i != paired_idx:
                        cuda.atomic.add(out_flat, paired_idx, inv_sqrt2 * (s0 - s1) - state_flat[paired_idx])
            
            elif gate_type == 5:  # CNOT (requires additional parameter)
                control = int(gate_data[gate_idx, 2].real)
                control_bit = n_qubits - control - 1
                target_bit = n_qubits - target - 1
                
                if (i >> control_bit) & 1:
                    swap_idx = i ^ (1 << target_bit)
                    temp = amplitude
                    amplitude = state_flat[swap_idx]
                    cuda.atomic.add(out_flat, swap_idx, temp - state_flat[swap_idx])
        
        out_flat[i] = amplitude
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
            gate_data[i, 1] = targets[1]  # target
            gate_data[i, 2] = targets[0]  # control
        else:
            return None  # Unsupported for template kernel
    
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
    
    try:
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
    except Exception:
        return False
