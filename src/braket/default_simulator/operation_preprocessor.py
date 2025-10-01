#!/usr/bin/env python3
"""
Advanced operation preprocessing for quantum circuit optimization.

This module implements sophisticated preprocessing techniques to drastically speed up
quantum simulations by:
1. Identity elimination and gate cancellation
2. Commutation-based reordering for better fusion opportunities
3. Phase gate consolidation and global phase extraction
4. Redundant operation detection and removal
5. Pattern-based optimization (e.g., Hadamard sandwich patterns)
6. Noise operation consolidation and early termination detection
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from braket.default_simulator.gate_operations import (
    CX,
    CY,
    CZ,
    GPhase,
    Hadamard,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    RotZ,
    S,
    Si,
    T,
    Ti,
)
from braket.default_simulator.noise_operations import BitFlip, PhaseFlip
from braket.default_simulator.operation import GateOperation, KrausOperation


@dataclass
class OptimizationStats:
    """Statistics about preprocessing optimizations applied."""

    original_count: int = 0
    final_count: int = 0
    identities_removed: int = 0
    cancellations: int = 0
    phase_consolidations: int = 0
    commutation_swaps: int = 0
    pattern_optimizations: int = 0
    noise_consolidations: int = 0

    @property
    def reduction_ratio(self) -> float:
        return self.final_count / self.original_count if self.original_count > 0 else 1.0

    @property
    def total_optimizations(self) -> int:
        return (
            self.identities_removed
            + self.cancellations
            + self.phase_consolidations
            + self.commutation_swaps
            + self.pattern_optimizations
            + self.noise_consolidations
        )


class OperationPreprocessor:
    """
    Advanced quantum circuit preprocessor for dramatic performance improvements.

    This preprocessor applies multiple optimization passes:
    1. Identity and cancellation elimination
    2. Phase gate consolidation
    3. Commutation-based reordering
    4. Pattern recognition and optimization
    5. Noise operation preprocessing
    """

    def __init__(self, max_passes: int = 5, enable_aggressive_opts: bool = True):
        self.max_passes = max_passes
        self.enable_aggressive_opts = enable_aggressive_opts
        self.stats = OptimizationStats()

        # Commutation rules: gates that commute with each other
        self._commutation_rules = self._build_commutation_rules()

        # Pattern recognition for common optimization opportunities
        self._optimization_patterns = self._build_optimization_patterns()

    def preprocess(
        self, operations: list[Union[GateOperation, KrausOperation]]
    ) -> list[Union[GateOperation, KrausOperation]]:
        """
        Apply comprehensive preprocessing optimizations.

        Args:
            operations: List of quantum operations to optimize

        Returns:
            Optimized list of operations
        """
        self.stats = OptimizationStats()
        self.stats.original_count = len(operations)

        if not operations:
            return operations

        # Make a copy to avoid modifying the original
        ops = list(operations)

        # Apply multiple optimization passes until convergence
        for pass_num in range(self.max_passes):
            initial_count = len(ops)

            # Pass 1: Remove identities and detect cancellations
            ops = self._remove_identities_and_cancellations(ops)

            # Pass 2: Consolidate phase gates and global phases
            ops = self._consolidate_phase_operations(ops)

            # Pass 3: Reorder operations for better fusion opportunities
            ops = self._commutation_based_reordering(ops)

            # Pass 4: Apply pattern-based optimizations
            if self.enable_aggressive_opts:
                ops = self._apply_pattern_optimizations(ops)

            # Pass 5: Preprocess noise operations
            ops = self._preprocess_noise_operations(ops)

            # Check for convergence
            if len(ops) == initial_count:
                break

        self.stats.final_count = len(ops)
        return ops

    def _remove_identities_and_cancellations(self, operations: list) -> list:
        """Remove identity operations and detect gate cancellations."""
        result = []
        i = 0

        while i < len(operations):
            op = operations[i]

            # Skip identity operations
            if isinstance(op, Identity):
                self.stats.identities_removed += 1
                i += 1
                continue

            # Check for gate cancellations (X-X, Y-Y, Z-Z, H-H, etc.)
            if i + 1 < len(operations):
                next_op = operations[i + 1]
                if self._operations_cancel(op, next_op):
                    self.stats.cancellations += 1
                    i += 2  # Skip both operations
                    continue

            # Check for more complex cancellation patterns (e.g., S-Si, T-Ti)
            if i + 1 < len(operations):
                next_op = operations[i + 1]
                if self._operations_are_inverses(op, next_op):
                    self.stats.cancellations += 1
                    i += 2
                    continue

            result.append(op)
            i += 1

        return result

    def _consolidate_phase_operations(self, operations: list) -> list:
        """Consolidate consecutive phase operations on the same qubits."""
        result = []
        phase_accumulator = defaultdict(float)  # qubit -> accumulated phase
        global_phase = 0.0

        i = 0
        while i < len(operations):
            op = operations[i]

            # Handle global phase operations
            if isinstance(op, GPhase) and len(op.targets) == 0:
                global_phase += getattr(op, "_angle", 0.0)
                self.stats.phase_consolidations += 1
                i += 1
                continue

            # Handle single-qubit phase operations
            if self._is_phase_operation(op) and len(op.targets) == 1:
                qubit = op.targets[0]
                phase = self._extract_phase(op)
                if phase is not None:
                    phase_accumulator[qubit] += phase
                    self.stats.phase_consolidations += 1
                    i += 1
                    continue

            # Flush accumulated phases before non-phase operations
            if phase_accumulator:
                for qubit, total_phase in phase_accumulator.items():
                    if abs(total_phase % (2 * np.pi)) > 1e-10:  # Only add non-trivial phases
                        result.append(PhaseShift(targets=[qubit], angle=total_phase))
                phase_accumulator.clear()

            result.append(op)
            i += 1

        # Flush remaining phases
        for qubit, total_phase in phase_accumulator.items():
            if abs(total_phase % (2 * np.pi)) > 1e-10:
                result.append(PhaseShift(targets=[qubit], angle=total_phase))

        # Add global phase if significant
        if abs(global_phase % (2 * np.pi)) > 1e-10:
            result.append(GPhase(targets=[], angle=global_phase))

        return result

    def _commutation_based_reordering(self, operations: list) -> list:
        """Reorder operations based on commutation rules to improve fusion opportunities."""
        if len(operations) <= 1:
            return operations

        result = []
        i = 0

        while i < len(operations):
            current_op = operations[i]

            # Look ahead for operations that can be reordered
            best_swap_pos = -1
            for j in range(i + 1, min(i + 10, len(operations))):  # Look ahead window
                candidate = operations[j]

                # Check if we can move candidate before current_op
                if self._can_commute_through(candidate, operations[i:j], current_op):
                    # Check if this improves fusion opportunities
                    if self._improves_fusion_opportunity(candidate, current_op):
                        best_swap_pos = j
                        break

            if best_swap_pos != -1:
                # Perform the swap
                swapped_op = operations[best_swap_pos]
                result.append(swapped_op)
                # Continue with the original operation next
                self.stats.commutation_swaps += 1
            else:
                result.append(current_op)
                i += 1

        return result

    def _apply_pattern_optimizations(self, operations: list) -> list:
        """Apply pattern-based optimizations for common gate sequences."""
        result = []
        i = 0

        while i < len(operations):
            # Check for optimization patterns
            pattern_applied = False

            for pattern_length in range(4, 1, -1):  # Check longer patterns first
                if i + pattern_length <= len(operations):
                    pattern = operations[i : i + pattern_length]
                    optimized = self._optimize_pattern(pattern)

                    if len(optimized) < len(pattern):
                        result.extend(optimized)
                        i += pattern_length
                        pattern_applied = True
                        self.stats.pattern_optimizations += 1
                        break

            if not pattern_applied:
                result.append(operations[i])
                i += 1

        return result

    def _preprocess_noise_operations(self, operations: list) -> list:
        """Preprocess noise operations for better performance."""
        result = []
        noise_groups = defaultdict(list)  # qubit -> list of consecutive noise ops

        for op in operations:
            if isinstance(op, KrausOperation):
                # Group consecutive noise operations by qubit
                if len(op.targets) == 1:
                    qubit = op.targets[0]
                    noise_groups[qubit].append(op)
                else:
                    # Flush existing groups and add multi-qubit noise
                    self._flush_noise_groups(noise_groups, result)
                    result.append(op)
            else:
                # Flush noise groups before gate operations
                self._flush_noise_groups(noise_groups, result)
                result.append(op)

        # Flush remaining noise groups
        self._flush_noise_groups(noise_groups, result)

        return result

    def _flush_noise_groups(self, noise_groups: dict, result: list):
        """Flush accumulated noise groups, applying consolidation."""
        for noise_ops in noise_groups.values():
            if len(noise_ops) > 1:
                # Try to consolidate similar noise operations
                consolidated = self._consolidate_noise_operations(noise_ops)
                result.extend(consolidated)
                self.stats.noise_consolidations += len(noise_ops) - len(consolidated)
            else:
                result.extend(noise_ops)
        noise_groups.clear()

    def _consolidate_noise_operations(
        self, noise_ops: list[KrausOperation]
    ) -> list[KrausOperation]:
        """Consolidate similar noise operations on the same qubit."""
        if not noise_ops:
            return []

        # Group by noise type
        by_type = defaultdict(list)
        for op in noise_ops:
            by_type[type(op)].append(op)

        result = []
        for noise_type, ops in by_type.items():
            if noise_type == BitFlip and len(ops) > 1:
                # Consolidate BitFlip operations: p_total = 1 - (1-p1)(1-p2)...
                total_prob = 1.0
                for op in ops:
                    total_prob *= 1.0 - op.probability
                final_prob = 1.0 - total_prob

                if final_prob > 1e-10:  # Only add if significant
                    result.append(BitFlip(probability=min(final_prob, 1.0), targets=ops[0].targets))
            elif noise_type == PhaseFlip and len(ops) > 1:
                # Similar consolidation for PhaseFlip
                total_prob = 1.0
                for op in ops:
                    total_prob *= 1.0 - op.probability
                final_prob = 1.0 - total_prob

                if final_prob > 1e-10:
                    result.append(
                        PhaseFlip(probability=min(final_prob, 1.0), targets=ops[0].targets)
                    )
            else:
                # For other noise types, keep all operations
                result.extend(ops)

        return result

    # Helper methods for optimization logic

    def _operations_cancel(self, op1, op2) -> bool:
        """Check if two operations cancel each other out."""
        if type(op1) is not type(op2) or op1.targets is not op2.targets:
            return False

        # Self-inverse gates
        self_inverse_types = (PauliX, PauliY, PauliZ, Hadamard, CX, CY, CZ)
        return isinstance(op1, self_inverse_types)

    def _operations_are_inverses(self, op1, op2) -> bool:
        """Check if two operations are inverses of each other."""
        if op1.targets != op2.targets:
            return False

        # Check specific inverse pairs
        inverse_pairs = [(S, Si), (Si, S), (T, Ti), (Ti, T)]

        for type1, type2 in inverse_pairs:
            if isinstance(op1, type1) and isinstance(op2, type2):
                return True

        return False

    def _is_phase_operation(self, op) -> bool:
        """Check if operation is a phase operation."""
        return isinstance(op, (PhaseShift, RotZ, S, Si, T, Ti, GPhase))

    def _extract_phase(self, op) -> Optional[float]:
        """Extract phase angle from a phase operation."""
        if isinstance(op, PhaseShift):
            return getattr(op, "_angle", 0.0)
        elif isinstance(op, RotZ):
            return getattr(op, "_angle", 0.0)
        elif isinstance(op, S):
            return np.pi / 2
        elif isinstance(op, Si):
            return -np.pi / 2
        elif isinstance(op, T):
            return np.pi / 4
        elif isinstance(op, Ti):
            return -np.pi / 4
        return None

    def _can_commute_through(self, op, intermediate_ops, target_op) -> bool:
        """Check if op can commute through intermediate_ops to reach target_op."""
        for intermediate in intermediate_ops:
            if not self._operations_commute(op, intermediate):
                return False
        return True

    def _operations_commute(self, op1, op2) -> bool:
        """Check if two operations commute."""
        # Operations on disjoint qubits always commute
        if not set(op1.targets).intersection(set(op2.targets)):
            return True

        # Use commutation rules
        return self._commutation_rules.get((type(op1), type(op2)), False)

    def _improves_fusion_opportunity(self, op1, op2) -> bool:
        """Check if reordering op1 before op2 improves fusion opportunities."""
        # Same type operations can potentially fuse
        if type(op1) is type(op2) and op1.targets is op2.targets:
            return True

        # Phase operations can consolidate
        if self._is_phase_operation(op1) and self._is_phase_operation(op2):
            return op1.targets == op2.targets

        return False

    def _optimize_pattern(self, pattern: list) -> list:
        """Optimize a specific pattern of operations."""
        if len(pattern) < 2:
            return pattern

        # Hadamard sandwich optimization: H-X-H = Z, H-Z-H = X
        if (
            len(pattern) == 3
            and isinstance(pattern[0], Hadamard)
            and isinstance(pattern[2], Hadamard)
            and pattern[0].targets == pattern[2].targets
        ):
            middle = pattern[1]
            if isinstance(middle, PauliX) and middle.targets == pattern[0].targets:
                return [PauliZ(targets=middle.targets)]
            elif isinstance(middle, PauliZ) and middle.targets == pattern[0].targets:
                return [PauliX(targets=middle.targets)]

        # X-Y-Z = -iI (up to global phase)
        if (
            len(pattern) == 3
            and isinstance(pattern[0], PauliX)
            and isinstance(pattern[1], PauliY)
            and isinstance(pattern[2], PauliZ)
            and pattern[0].targets == pattern[1].targets == pattern[2].targets
        ):
            return [GPhase(targets=[], angle=-np.pi / 2)]

        return pattern

    def _build_commutation_rules(self) -> dict[tuple, bool]:
        """Build commutation rules for gate types."""
        rules = {}

        # Pauli gates commute with themselves
        pauli_types = (PauliX, PauliY, PauliZ)
        for p1 in pauli_types:
            for p2 in pauli_types:
                rules[(p1, p2)] = True
                rules[(p2, p1)] = True

        # Phase gates commute with Z and other phase gates
        phase_types = (PhaseShift, RotZ, S, Si, T, Ti)
        for p1 in phase_types:
            for p2 in phase_types:
                rules[(p1, p2)] = True
                rules[(p2, p1)] = True
            rules[(p1, PauliZ)] = True
            rules[(PauliZ, p1)] = True

        return rules

    def _build_optimization_patterns(self) -> dict:
        """Build optimization patterns for common gate sequences."""
        # This could be expanded with more sophisticated pattern matching
        return {}

    def get_stats(self) -> OptimizationStats:
        """Get optimization statistics."""
        return self.stats


def preprocess_operations(
    operations: list[Union[GateOperation, KrausOperation]], aggressive: bool = True
) -> tuple[list[Union[GateOperation, KrausOperation]], OptimizationStats]:
    """
    Convenience function to preprocess quantum operations with advanced optimizations.

    Args:
        operations: List of quantum operations to optimize
        aggressive: Whether to enable aggressive optimizations

    Returns:
        Tuple of (optimized_operations, optimization_stats)
    """
    preprocessor = OperationPreprocessor(enable_aggressive_opts=aggressive)
    optimized_ops = preprocessor.preprocess(operations)
    return optimized_ops, preprocessor.get_stats()
