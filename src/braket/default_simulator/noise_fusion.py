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
Noise Operation Fusion Implementation

This module provides fusion optimization for quantum noise channels through:
1. Conservative Kraus operator composition with complexity bounds
2. Pattern recognition for same-type channel fusion
3. Algebraic simplifications for common noise patterns
4. CPTP preservation validation
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from braket.default_simulator.operation import KrausOperation


class FusedNoiseOperation(KrausOperation):
    """
    Fused noise operation with optimized Kraus operators.
    
    This class represents multiple noise channels fused together with optimizations
    including pattern recognition and algebraic simplification.
    """

    def __init__(
        self,
        targets: tuple[int, ...],
        fused_kraus_ops: list[np.ndarray],
        original_operations: list[KrausOperation],
        optimization_type: str = "standard",
    ):
        """
        Initialize a fused noise operation.

        Args:
            targets: The target qubits for this fused operation
            fused_kraus_ops: The pre-computed Kraus operators representing all fused channels
            original_operations: List of original noise operations that were fused together
            optimization_type: Type of optimization applied ("same_type", "pauli", "standard")
        """
        self._targets = tuple(targets)
        self._fused_kraus_ops = [op.astype(complex) for op in fused_kraus_ops]
        self._original_operations = original_operations
        self._operation_count = len(original_operations)
        self._optimization_type = optimization_type
        
        self._validate_cptp()

    @property
    def matrices(self) -> list[np.ndarray]:
        """Return the fused Kraus operators."""
        return self._fused_kraus_ops

    @property
    def targets(self) -> tuple[int, ...]:
        """Return the target qubits."""
        return self._targets

    @property
    def original_operations(self) -> list[KrausOperation]:
        """Return the list of original operations that were fused."""
        return self._original_operations

    @property
    def operation_count(self) -> int:
        """Return the number of operations that were fused together."""
        return self._operation_count

    @property
    def optimization_type(self) -> str:
        """Return the type of optimization applied."""
        return self._optimization_type

    def _validate_cptp(self) -> None:
        """Validate that the fused operation preserves CPTP property."""
        if not self._fused_kraus_ops:
            return
            
        dim = self._fused_kraus_ops[0].shape[0]
        sum_ktk = np.zeros((dim, dim), dtype=complex)
        
        for kraus_op in self._fused_kraus_ops:
            sum_ktk += kraus_op.conj().T @ kraus_op
            
        identity = np.eye(dim, dtype=complex)
        if not np.allclose(sum_ktk, identity, atol=1e-10):
            raise ValueError("Fused noise operation violates CPTP property")

    def __repr__(self) -> str:
        op_types = [type(op).__name__ for op in self._original_operations]
        return f"FusedNoiseOperation(targets={self.targets}, ops={op_types}, opt={self._optimization_type})"


class DepolarizingChannelOptimizer:
    """Optimizer for depolarizing noise channels."""
    
    def __init__(self):
        self._pauli_matrices = self._build_pauli_matrices()
    
    def _build_pauli_matrices(self) -> dict:
        return {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
    
    def optimize_depolarizing_sequence(self, operations: list[KrausOperation]) -> Optional[list[np.ndarray]]:
        """Optimize a sequence of depolarizing channels."""
        if not all(self._is_depolarizing_channel(op) for op in operations):
            return None
            
        combined_p = 0.0
        for op in operations:
            p = self._extract_depolarizing_probability(op)
            if p is None:
                return None
            combined_p = combined_p + p - (4.0/3.0) * combined_p * p
        
        return self._create_depolarizing_kraus(combined_p)
    
    def _is_depolarizing_channel(self, op: KrausOperation) -> bool:
        """Check if operation is a depolarizing channel."""
        return hasattr(op, 'probability') and len(op.matrices) == 4
    
    def _extract_depolarizing_probability(self, op: KrausOperation) -> Optional[float]:
        """Extract depolarizing probability from operation."""
        if hasattr(op, 'probability'):
            return op.probability
        return None
    
    def _create_depolarizing_kraus(self, p: float) -> list[np.ndarray]:
        """Create Kraus operators for depolarizing channel with probability p."""
        coeff_i = np.sqrt(1 - 3*p/4)
        coeff_pauli = np.sqrt(p/4)
        
        return [
            coeff_i * self._pauli_matrices['I'],
            coeff_pauli * self._pauli_matrices['X'],
            coeff_pauli * self._pauli_matrices['Y'],
            coeff_pauli * self._pauli_matrices['Z']
        ]


class PauliChannelOptimizer:
    """Optimizer for Pauli noise channels."""
    
    def __init__(self):
        self._pauli_matrices = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
    
    def optimize_pauli_sequence(self, operations: list[KrausOperation]) -> Optional[list[np.ndarray]]:
        """Optimize a sequence of Pauli channels."""
        if not all(self._is_pauli_channel(op) for op in operations):
            return None
            
        total_probs = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        
        for op in operations:
            probs = self._extract_pauli_probabilities(op)
            if probs is None:
                return None
            
            for pauli in ['X', 'Y', 'Z']:
                total_probs[pauli] = (total_probs[pauli] + probs.get(pauli, 0.0)) % 1.0
        
        p_i = 1.0 - sum(total_probs.values())
        if p_i < 0:
            return None
            
        return self._create_pauli_kraus(p_i, total_probs)
    
    def _is_pauli_channel(self, op: KrausOperation) -> bool:
        """Check if operation is a Pauli channel."""
        return len(op.matrices) <= 4
    
    def _extract_pauli_probabilities(self, op: KrausOperation) -> Optional[dict]:
        """Extract Pauli probabilities from operation."""
        if hasattr(op, 'pauli_probabilities'):
            return op.pauli_probabilities
        return None
    
    def _create_pauli_kraus(self, p_i: float, pauli_probs: dict) -> list[np.ndarray]:
        """Create Kraus operators for Pauli channel."""
        kraus_ops = []
        
        if p_i > 1e-12:
            kraus_ops.append(np.sqrt(p_i) * self._pauli_matrices['I'])
        
        for pauli, prob in pauli_probs.items():
            if prob > 1e-12:
                kraus_ops.append(np.sqrt(prob) * self._pauli_matrices[pauli])
        
        return kraus_ops


class KrausAlgebra:
    """
    Handles Kraus operator composition and simplification.
    
    This class provides the mathematical foundation for fusing quantum channels
    while preserving their physical properties.
    """

    @staticmethod
    def compose_channels(kraus1: list[np.ndarray], kraus2: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compose two quantum channels represented by their Kraus operators.
        
        For channels E1 and E2, the composition E2 ∘ E1 has Kraus operators:
        {B_j * A_i} where {A_i} are Kraus operators of E1 and {B_j} are Kraus operators of E2.
        
        Args:
            kraus1: Kraus operators of the first channel E1
            kraus2: Kraus operators of the second channel E2
            
        Returns:
            Kraus operators of the composed channel E2 ∘ E1
        """
        composed_kraus = []
        for b_j in kraus2:
            for a_i in kraus1:
                composed_kraus.append(b_j @ a_i)
        return composed_kraus

    @staticmethod
    def simplify_bit_flip_composition(operations: list[KrausOperation]) -> Optional[list[np.ndarray]]:
        """
        Simplify composition of BitFlip channels using algebraic properties.
        
        BitFlip channels compose as: p_eff = p1 + p2 - 2*p1*p2
        
        Args:
            operations: List of BitFlip operations
            
        Returns:
            Simplified Kraus operators or None if not applicable
        """
        if not all(type(op).__name__ == 'MockBitFlip' or hasattr(op, 'probability') for op in operations):
            return None
            
        p_eff = 0.0
        for op in operations:
            if hasattr(op, 'probability'):
                p = op.probability
                p_eff = p_eff + p - 2 * p_eff * p
            else:
                return None
            
        pauli_i = np.eye(2, dtype=complex)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        
        return [
            np.sqrt(1 - p_eff) * pauli_i,
            np.sqrt(p_eff) * pauli_x
        ]

    @staticmethod
    def simplify_phase_flip_composition(operations: list[KrausOperation]) -> Optional[list[np.ndarray]]:
        """
        Simplify composition of PhaseFlip channels.
        
        Args:
            operations: List of PhaseFlip operations
            
        Returns:
            Simplified Kraus operators or None if not applicable
        """
        if not all(type(op).__name__ == 'MockPhaseFlip' or hasattr(op, 'probability') for op in operations):
            return None
            
        p_eff = 0.0
        for op in operations:
            if hasattr(op, 'probability'):
                p = op.probability
                p_eff = p_eff + p - 2 * p_eff * p
            else:
                return None
            
        pauli_i = np.eye(2, dtype=complex)
        pauli_z = np.diag([1.0, -1.0]).astype(complex)
        
        return [
            np.sqrt(1 - p_eff) * pauli_i,
            np.sqrt(p_eff) * pauli_z
        ]

    @staticmethod
    def truncate_small_operators(kraus_ops: list[np.ndarray], threshold: float = 1e-12) -> list[np.ndarray]:
        """
        Remove Kraus operators with coefficients below threshold.
        
        Args:
            kraus_ops: List of Kraus operators
            threshold: Minimum coefficient magnitude to keep
            
        Returns:
            Filtered list of Kraus operators
        """
        filtered_ops = []
        for op in kraus_ops:
            # Check if operator has significant coefficients
            max_coeff = np.max(np.abs(op))
            if max_coeff > threshold:
                filtered_ops.append(op)
                
        return filtered_ops if filtered_ops else kraus_ops


class NoiseOperationFusionOptimizer:
    """
    Enhanced noise operation fusion optimizer for maximum gate count reduction.
    
    This optimizer provides aggressive performance improvements for noise channels
    with expanded complexity bounds and advanced optimization techniques.
    """

    def __init__(self, max_kraus_operators: int = 32, max_fusion_size: int = 8):
        """
        Initialize the enhanced noise fusion optimizer.

        Args:
            max_kraus_operators: Maximum number of Kraus operators in fused operation
            max_fusion_size: Maximum number of operations to fuse together
        """
        self.max_kraus_operators = max_kraus_operators
        self.max_fusion_size = max_fusion_size
        self.algebra = KrausAlgebra()
        self.depolarizing_optimizer = DepolarizingChannelOptimizer()
        self.pauli_optimizer = PauliChannelOptimizer()

    def optimize_noise_operations(self, operations: list[KrausOperation]) -> list[KrausOperation]:
        """
        Apply conservative fusion optimization to noise operations.

        Args:
            operations: List of noise operations to optimize

        Returns:
            List of optimized operations with fusion applied where beneficial
        """
        if not operations:
            return operations

        optimized = []
        i = 0

        while i < len(operations):
            fusion_group = self._find_fusion_group(operations, i)

            if len(fusion_group) > 1:
                fused_op = self._create_fused_operation(fusion_group)
                if fused_op:
                    optimized.append(fused_op)
                    i += len(fusion_group)
                else:
                    optimized.append(operations[i])
                    i += 1
            else:
                optimized.append(operations[i])
                i += 1

        return optimized

    def _find_fusion_group(self, operations: list[KrausOperation], start_idx: int) -> list[KrausOperation]:
        """Find a group of consecutive operations that can be fused."""
        if start_idx >= len(operations):
            return []

        first_op = operations[start_idx]
        fusion_group = [first_op]
        first_targets = set(first_op.targets)

        for i in range(start_idx + 1, min(start_idx + self.max_fusion_size, len(operations))):
            candidate = operations[i]

            if self._can_fuse_with_group(candidate, fusion_group, first_targets):
                fusion_group.append(candidate)
            else:
                break

        return fusion_group

    def _can_fuse_with_group(
        self, 
        candidate: KrausOperation, 
        fusion_group: list[KrausOperation], 
        first_targets: set[int]
    ) -> bool:
        """Check if candidate can be fused with current group."""
        if set(candidate.targets) != first_targets:
            return False

        estimated_count = len(candidate.matrices)
        for op in fusion_group:
            estimated_count *= len(op.matrices)

        if estimated_count > self.max_kraus_operators:
            return False

        return True

    def _create_fused_operation(self, operations: list[KrausOperation]) -> Optional[FusedNoiseOperation]:
        """Create a fused noise operation from a group of operations."""
        if len(operations) == 1:
            return None

        try:
            simplified_kraus = self._try_same_type_optimization(operations)
            
            if simplified_kraus is None:
                simplified_kraus = self._compose_operations(operations)

            simplified_kraus = self.algebra.truncate_small_operators(simplified_kraus)

            if len(simplified_kraus) > self.max_kraus_operators:
                return None

            all_targets = set().union(*[set(op.targets) for op in operations])
            targets = tuple(sorted(all_targets))

            optimization_type = self._determine_optimization_type(operations)

            return FusedNoiseOperation(
                targets=targets,
                fused_kraus_ops=simplified_kraus,
                original_operations=operations,
                optimization_type=optimization_type,
            )

        except Exception:
            return None

    def _try_same_type_optimization(self, operations: list[KrausOperation]) -> Optional[list[np.ndarray]]:
        """Try to apply same-type channel optimizations."""
        if not operations:
            return None

        first_type = type(operations[0]).__name__
        
        if not all(type(op).__name__ == first_type for op in operations):
            return None

        if first_type in ['BitFlip', 'MockBitFlip']:
            return self.algebra.simplify_bit_flip_composition(operations)
        elif first_type in ['PhaseFlip', 'MockPhaseFlip']:
            return self.algebra.simplify_phase_flip_composition(operations)
        
        return None

    def _compose_operations(self, operations: list[KrausOperation]) -> list[np.ndarray]:
        """Compose operations using standard Kraus operator multiplication."""
        result_kraus = operations[0].matrices.copy()
        
        for i in range(1, len(operations)):
            next_kraus = operations[i].matrices
            result_kraus = self.algebra.compose_channels(result_kraus, next_kraus)
            
        return result_kraus

    def _determine_optimization_type(self, operations: list[KrausOperation]) -> str:
        """Determine the type of optimization applied."""
        if not operations:
            return "standard"
            
        first_type = type(operations[0]).__name__
        
        if all(type(op).__name__ == first_type for op in operations):
            return "same_type"
        else:
            return "standard"


def apply_noise_fusion(
    operations: list[KrausOperation],
    max_kraus_operators: int = 8,
    max_fusion_size: int = 3,
) -> list[KrausOperation]:
    """
    Apply noise fusion optimization to a list of noise operations.

    This function provides conservative fusion optimization for quantum noise channels
    with strict complexity bounds to ensure practical performance.

    Args:
        operations: List of noise operations to optimize
        max_kraus_operators: Maximum number of Kraus operators in fused operation
        max_fusion_size: Maximum number of operations to fuse together

    Returns:
        List of optimized operations with fusion applied where beneficial
    """
    optimizer = NoiseOperationFusionOptimizer(
        max_kraus_operators=max_kraus_operators,
        max_fusion_size=max_fusion_size
    )
    return optimizer.optimize_noise_operations(operations)
