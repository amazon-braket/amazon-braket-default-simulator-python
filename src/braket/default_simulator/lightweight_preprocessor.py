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

from braket.default_simulator.operation import GateOperation, KrausOperation
from braket.default_simulator.gate_operations import Identity, PauliX, PauliY, PauliZ, Hadamard, S, Si, T, Ti, PhaseShift, GPhase
from braket.default_simulator.noise_operations import BitFlip, PhaseFlip


class LightweightPreprocessor:
    """
    Fast, lightweight preprocessor focusing on high-impact optimizations.
    
    This preprocessor applies only the most beneficial optimizations with minimal overhead:
    - Identity elimination (O(n))
    - Gate cancellation (O(n))  
    - Phase consolidation (O(n))
    - Basic noise consolidation (O(n))
    
    All operations are single-pass for maximum speed.
    """
    
    def __init__(self):
        pass
    
    def preprocess(self, operations: list[GateOperation | KrausOperation]) -> list[GateOperation | KrausOperation]:
        """
        Apply fast preprocessing optimizations.
        
        Args:
            operations: List of quantum operations to optimize
            
        Returns:
            Optimized list of operations
        """
        if not operations:
            return operations
        
        result = []
        i = 0
        
        while i < len(operations):
            op = operations[i]
            
            if isinstance(op, Identity):
                i += 1
                continue
            
            if i + 1 < len(operations):
                next_op = operations[i + 1]
                if self._operations_cancel(op, next_op):
                    i += 2
                    continue
            
            if self._is_phase_operation(op):
                consolidated_op, consumed = self._consolidate_phases(operations, i)
                if consumed > 1:
                    if consolidated_op is not None:
                        result.append(consolidated_op)
                    i += consumed
                    continue
            
            if isinstance(op, (BitFlip, PhaseFlip)):
                consolidated_op, consumed = self._consolidate_noise(operations, i)
                if consumed > 1:
                    result.append(consolidated_op)
                    i += consumed
                    continue
            
            result.append(op)
            i += 1
        
        return result
    
    def _operations_cancel(self, op1, op2) -> bool:
        """Check if two operations cancel each other out."""
        if type(op1) != type(op2) or op1.targets != op2.targets:
            return False
        
        self_inverse_types = (PauliX, PauliY, PauliZ, Hadamard)
        if isinstance(op1, self_inverse_types):
            return True
        
        if (isinstance(op1, S) and isinstance(op2, Si)) or (isinstance(op1, Si) and isinstance(op2, S)):
            return True
        if (isinstance(op1, T) and isinstance(op2, Ti)) or (isinstance(op1, Ti) and isinstance(op2, T)):
            return True
        
        return False
    
    def _is_phase_operation(self, op) -> bool:
        """Check if operation is a phase operation."""
        return isinstance(op, (PhaseShift, S, Si, T, Ti, GPhase))
    
    def _consolidate_phases(self, operations, start_idx):
        """Consolidate consecutive phase operations on the same qubit."""
        first_op = operations[start_idx]
        
        if isinstance(first_op, GPhase) and len(first_op.targets) == 0:
            total_phase = getattr(first_op, '_angle', 0.0)
            consumed = 1
            
            for i in range(start_idx + 1, len(operations)):
                op = operations[i]
                if isinstance(op, GPhase) and len(op.targets) == 0:
                    total_phase += getattr(op, '_angle', 0.0)
                    consumed += 1
                else:
                    break
            
            if abs(total_phase % (2 * np.pi)) > 1e-10:
                return GPhase(targets=[], angle=total_phase), consumed
            else:
                return None, consumed
        
        if len(first_op.targets) != 1:
            return first_op, 1
        
        qubit = first_op.targets[0]
        total_phase = self._extract_phase(first_op)
        if total_phase is None:
            return first_op, 1
        
        consumed = 1
        
        for i in range(start_idx + 1, len(operations)):
            op = operations[i]
            if (self._is_phase_operation(op) and len(op.targets) == 1 and 
                op.targets[0] == qubit):
                phase = self._extract_phase(op)
                if phase is not None:
                    total_phase += phase
                    consumed += 1
                else:
                    break
            else:
                break
        
        if consumed > 1:
            if abs(total_phase % (2 * np.pi)) > 1e-10:
                return PhaseShift(targets=[qubit], angle=total_phase), consumed
            else:
                return None, consumed
        
        return first_op, 1
    
    def _extract_phase(self, op):
        """Extract phase angle from a phase operation."""
        if isinstance(op, PhaseShift):
            return getattr(op, '_angle', 0.0)
        elif isinstance(op, S):
            return np.pi / 2
        elif isinstance(op, Si):
            return -np.pi / 2
        elif isinstance(op, T):
            return np.pi / 4
        elif isinstance(op, Ti):
            return -np.pi / 4
        return None
    
    def _consolidate_noise(self, operations, start_idx):
        """Consolidate consecutive noise operations of the same type on the same qubit."""
        first_op = operations[start_idx]
        
        if len(first_op.targets) != 1:
            return first_op, 1
        
        qubit = first_op.targets[0]
        noise_type = type(first_op)
        total_prob = first_op.probability
        consumed = 1
        
        for i in range(start_idx + 1, len(operations)):
            op = operations[i]
            if (isinstance(op, noise_type) and len(op.targets) == 1 and 
                op.targets[0] == qubit):
                total_prob = 1.0 - (1.0 - total_prob) * (1.0 - op.probability)
                consumed += 1
            else:
                break
        
        if consumed > 1:
            return noise_type(probability=min(total_prob, 1.0), targets=[qubit]), consumed
        
        return first_op, 1
    


def fast_preprocess(operations: list[GateOperation | KrausOperation]) -> list[GateOperation | KrausOperation]:
    """
    Fast preprocessing function for quantum operations.
    
    Args:
        operations: List of quantum operations to optimize
        
    Returns:
        Optimized list of operations
    """
    if not operations:
        return operations
    
    preprocessor = LightweightPreprocessor()
    return preprocessor.preprocess(operations)
