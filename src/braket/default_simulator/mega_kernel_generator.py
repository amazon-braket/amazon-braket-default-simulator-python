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
import hashlib
import tempfile
import subprocess
import os

from braket.default_simulator.operation import GateOperation
from braket.default_simulator.linalg_utils import (
    DIAGONAL_GATES,
    _GPU_AVAILABLE,
    _MAX_BLOCKS_PER_GRID,
)


class MegaKernelGenerator:
    """Generates optimized mega-kernels that fuse entire quantum circuits into single CUDA kernels."""
    
    def __init__(self):
        self.compiled_kernels = {}
        self.kernel_cache = {}
        
    def can_generate_mega_kernel(self, operations: list[GateOperation]) -> bool:
        """Determine if circuit can be fused into a single mega-kernel."""
        if len(operations) < 3 or len(operations) > 50:
            return False
        
        register_usage = 0
        for op in operations:
            num_targets = len(op.targets)
            num_controls = len(getattr(op, '_ctrl_modifiers', []))
            
            if num_targets > 2 or num_controls > 1:
                return False
            
            register_usage += 4 + num_targets * 2
            if register_usage > 40:
                return False
        
        return True
    
    def generate_mega_kernel_code(self, operations: list[GateOperation], qubit_count: int) -> str:
        """Generate CUDA C code for mega-kernel that processes entire circuit."""
        operation_code_blocks = []
        
        for i, op in enumerate(operations):
            op_code = self._generate_operation_cuda_code(op, i, qubit_count)
            if op_code:
                operation_code_blocks.append(f"        // Operation {i}: {getattr(op, 'gate_type', 'custom')}")
                operation_code_blocks.append(op_code)
            else:
                return None
        
        mega_kernel_code = f"""
extern "C" __global__ void mega_circuit_kernel(
    cuDoubleComplex* state_flat,
    int total_size
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < total_size) {{
        cuDoubleComplex amplitude = state_flat[i];
        
{chr(10).join(operation_code_blocks)}
        
        state_flat[i] = amplitude;
        i += stride;
    }}
}}
"""
        
        return mega_kernel_code
    
    def _generate_operation_cuda_code(self, op: GateOperation, op_index: int, qubit_count: int) -> str:
        """Generate CUDA code for a single quantum operation."""
        gate_type = getattr(op, 'gate_type', None)
        targets = op.targets
        
        if len(targets) == 1 and not getattr(op, '_ctrl_modifiers', []):
            return self._generate_single_qubit_code(op, op_index, qubit_count)
        elif len(targets) == 2 and not getattr(op, '_ctrl_modifiers', []):
            return self._generate_two_qubit_code(op, op_index, qubit_count)
        elif len(getattr(op, '_ctrl_modifiers', [])) == 1:
            return self._generate_controlled_code(op, op_index, qubit_count)
        else:
            return None
    
    def _generate_single_qubit_code(self, op: GateOperation, op_index: int, qubit_count: int) -> str:
        """Generate optimized single-qubit operation code."""
        target = op.targets[0]
        target_bit = qubit_count - target - 1
        gate_type = getattr(op, 'gate_type', None)
        
        if gate_type == 'pauli_x':
            return f"""
        {{
            int paired_{op_index} = i ^ (1 << {target_bit});
            if (i <= paired_{op_index}) {{
                cuDoubleComplex temp = amplitude;
                amplitude = state_flat[paired_{op_index}];
                if (i != paired_{op_index}) state_flat[paired_{op_index}] = temp;
            }}
        }}"""
        
        elif gate_type == 'pauli_z':
            return f"""
        {{
            if ((i >> {target_bit}) & 1) amplitude = cuCmul(amplitude, make_cuDoubleComplex(-1.0, 0.0));
        }}"""
        
        elif gate_type == 'h':
            return f"""
        {{
            int paired_{op_index} = i ^ (1 << {target_bit});
            if (i <= paired_{op_index}) {{
                cuDoubleComplex s0 = amplitude;
                cuDoubleComplex s1 = (i == paired_{op_index}) ? amplitude : state_flat[paired_{op_index}];
                double inv_sqrt2 = 0.7071067811865476;
                
                amplitude = cuCmul(make_cuDoubleComplex(inv_sqrt2, 0.0), cuCadd(s0, s1));
                if (i != paired_{op_index}) {{
                    state_flat[paired_{op_index}] = cuCmul(make_cuDoubleComplex(inv_sqrt2, 0.0), cuCsub(s0, s1));
                }}
            }}
        }}"""
        
        else:
            matrix = op.matrix
            a, b, c, d = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
            
            return f"""
        {{
            int paired_{op_index} = i ^ (1 << {target_bit});
            if (i <= paired_{op_index}) {{
                cuDoubleComplex s0 = amplitude;
                cuDoubleComplex s1 = (i == paired_{op_index}) ? amplitude : state_flat[paired_{op_index}];
                
                amplitude = cuCadd(cuCmul(make_cuDoubleComplex({a.real}, {a.imag}), s0),
                                 cuCmul(make_cuDoubleComplex({b.real}, {b.imag}), s1));
                if (i != paired_{op_index}) {{
                    state_flat[paired_{op_index}] = cuCadd(cuCmul(make_cuDoubleComplex({c.real}, {c.imag}), s0),
                                                          cuCmul(make_cuDoubleComplex({d.real}, {d.imag}), s1));
                }}
            }}
        }}"""
    
    def _generate_two_qubit_code(self, op: GateOperation, op_index: int, qubit_count: int) -> str:
        """Generate optimized two-qubit operation code."""
        target0, target1 = op.targets[0], op.targets[1]
        gate_type = getattr(op, 'gate_type', None)
        
        if gate_type == 'cx':
            control_bit = qubit_count - target0 - 1
            target_bit = qubit_count - target1 - 1
            
            return f"""
        {{
            if ((i >> {control_bit}) & 1) {{
                int swap_{op_index} = i ^ (1 << {target_bit});
                cuDoubleComplex temp = amplitude;
                amplitude = state_flat[swap_{op_index}];
                state_flat[swap_{op_index}] = temp;
            }}
        }}"""
        
        else:
            mask_0 = 1 << (qubit_count - 1 - target0)
            mask_1 = 1 << (qubit_count - 1 - target1)
            mask_both = mask_0 | mask_1
            matrix = op.matrix
            
            matrix_vals = []
            for row in range(4):
                for col in range(4):
                    val = matrix[row, col]
                    matrix_vals.append(f"make_cuDoubleComplex({val.real}, {val.imag})")
            
            return f"""
        {{
            if ((i & {mask_both}) == 0) {{
                cuDoubleComplex s0 = amplitude;
                cuDoubleComplex s1 = state_flat[i | {mask_1}];
                cuDoubleComplex s2 = state_flat[i | {mask_0}];
                cuDoubleComplex s3 = state_flat[i | {mask_both}];
                
                cuDoubleComplex m0 = {matrix_vals[0]}, m1 = {matrix_vals[1]}, m2 = {matrix_vals[2]}, m3 = {matrix_vals[3]};
                cuDoubleComplex m4 = {matrix_vals[4]}, m5 = {matrix_vals[5]}, m6 = {matrix_vals[6]}, m7 = {matrix_vals[7]};
                cuDoubleComplex m8 = {matrix_vals[8]}, m9 = {matrix_vals[9]}, m10 = {matrix_vals[10]}, m11 = {matrix_vals[11]};
                cuDoubleComplex m12 = {matrix_vals[12]}, m13 = {matrix_vals[13]}, m14 = {matrix_vals[14]}, m15 = {matrix_vals[15]};
                
                amplitude = cuCadd(cuCadd(cuCmul(m0, s0), cuCmul(m1, s1)), cuCadd(cuCmul(m2, s2), cuCmul(m3, s3)));
                state_flat[i | {mask_1}] = cuCadd(cuCadd(cuCmul(m4, s0), cuCmul(m5, s1)), cuCadd(cuCmul(m6, s2), cuCmul(m7, s3)));
                state_flat[i | {mask_0}] = cuCadd(cuCadd(cuCmul(m8, s0), cuCmul(m9, s1)), cuCadd(cuCmul(m10, s2), cuCmul(m11, s3)));
                state_flat[i | {mask_both}] = cuCadd(cuCadd(cuCmul(m12, s0), cuCmul(m13, s1)), cuCadd(cuCmul(m14, s2), cuCmul(m15, s3)));
            }}
        }}"""
    
    def _generate_controlled_code(self, op: GateOperation, op_index: int, qubit_count: int) -> str:
        """Generate single-controlled gate code."""
        control = op.targets[0]
        target = op.targets[1]
        control_bit = qubit_count - control - 1
        target_bit = qubit_count - target - 1
        
        controlled_matrix = op.matrix[2:, 2:]
        a, b, c, d = controlled_matrix[0,0], controlled_matrix[0,1], controlled_matrix[1,0], controlled_matrix[1,1]
        
        return f"""
        {{
            if ((i >> {control_bit}) & 1) {{
                int paired_{op_index} = i ^ (1 << {target_bit});
                if (i <= paired_{op_index}) {{
                    cuDoubleComplex s0 = amplitude;
                    cuDoubleComplex s1 = (i == paired_{op_index}) ? amplitude : state_flat[paired_{op_index}];
                    
                    amplitude = cuCadd(cuCmul(make_cuDoubleComplex({a.real}, {a.imag}), s0),
                                     cuCmul(make_cuDoubleComplex({b.real}, {b.imag}), s1));
                    if (i != paired_{op_index}) {{
                        state_flat[paired_{op_index}] = cuCadd(cuCmul(make_cuDoubleComplex({c.real}, {c.imag}), s0),
                                                              cuCmul(make_cuDoubleComplex({d.real}, {d.imag}), s1));
                    }}
                }}
            }}
        }}"""
    
    def compile_mega_kernel(self, kernel_code: str, circuit_signature: str):
        """Compile mega-kernel CUDA code into executable function."""
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
        kernel = module.get_function('mega_circuit_kernel')
        
        for file_path in [cu_file, ptx_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        self.compiled_kernels[circuit_signature] = kernel
        return kernel
    
    def execute_mega_kernel(self, kernel, state_gpu):
        """Execute mega-kernel with optimized in-place processing."""
        total_size = state_gpu.size
        state_flat = state_gpu.reshape(-1)
        
        threads_per_block = 1024
        blocks_per_grid = min(
            (total_size + threads_per_block - 1) // threads_per_block,
            _MAX_BLOCKS_PER_GRID
        )
        
        kernel[blocks_per_grid, threads_per_block](state_flat, total_size)
    
    def get_circuit_signature(self, operations: list[GateOperation], qubit_count: int) -> str:
        """Generate unique signature for mega-kernel caching."""
        sig_parts = [f"qubits:{qubit_count}"]
        
        for i, op in enumerate(operations):
            gate_type = getattr(op, 'gate_type', 'custom')
            targets = ','.join(map(str, op.targets))
            controls = ','.join(map(str, getattr(op, '_ctrl_modifiers', [])))
            matrix_hash = hash(op.matrix.tobytes()) % 10000 if hasattr(op, 'matrix') else 0
            
            sig_parts.append(f"op{i}:{gate_type}:t{targets}:c{controls}:m{matrix_hash}")
        
        return hashlib.md5('|'.join(sig_parts).encode()).hexdigest()


_mega_kernel_generator = MegaKernelGenerator() if _GPU_AVAILABLE else None


def execute_mega_kernel_circuit(
    operations: list[GateOperation],
    state_gpu: cuda.devicearray.DeviceNDArray,
    qubit_count: int
) -> bool:
    """Execute quantum circuit using mega-kernel optimization."""
    if not _mega_kernel_generator:
        return False
    
    if not _mega_kernel_generator.can_generate_mega_kernel(operations):
        return False
    
    circuit_signature = _mega_kernel_generator.get_circuit_signature(operations, qubit_count)
    
    if circuit_signature in _mega_kernel_generator.compiled_kernels:
        kernel = _mega_kernel_generator.compiled_kernels[circuit_signature]
    else:
        kernel_code = _mega_kernel_generator.generate_mega_kernel_code(operations, qubit_count)
        if not kernel_code:
            return False
        
        kernel = _mega_kernel_generator.compile_mega_kernel(kernel_code, circuit_signature)
        if not kernel:
            return False
    
    _mega_kernel_generator.execute_mega_kernel(kernel, state_gpu)
    return True
