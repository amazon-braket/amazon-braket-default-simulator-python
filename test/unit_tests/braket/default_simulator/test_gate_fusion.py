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
import pytest

from braket.default_simulator.gate_fusion import (
    FusedGate,
    adaptive_optimize,
    estimate_fusion_benefit,
    fuse_adjacent_gates,
    fuse_operations,
    fuse_single_qubit_gates,
    fuse_two_qubit_gates,
    optimize_circuit,
)
from braket.default_simulator.gate_operations import CX, CZ, Hadamard, PauliX, RotX, RotZ, S, T


class TestFuseSingleQubitGates:
    def test_no_fusion_single_op(self):
        ops = [Hadamard([0])]
        result = fuse_single_qubit_gates(ops)
        assert len(result) == 1
        assert result[0] is ops[0]

    def test_fuse_consecutive_same_target(self):
        ops = [Hadamard([0]), T([0]), S([0])]
        result = fuse_single_qubit_gates(ops)
        assert len(result) == 1
        assert isinstance(result[0], FusedGate)
        assert result[0].targets == (0,)

    def test_no_fusion_different_targets(self):
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2])]
        result = fuse_single_qubit_gates(ops)
        assert len(result) == 3

    def test_partial_fusion(self):
        ops = [Hadamard([0]), T([0]), Hadamard([1]), S([1])]
        result = fuse_single_qubit_gates(ops)
        assert len(result) == 2
        assert isinstance(result[0], FusedGate)
        assert isinstance(result[1], FusedGate)

    def test_two_qubit_gate_breaks_fusion(self):
        ops = [Hadamard([0]), CX([0, 1]), T([0])]
        result = fuse_single_qubit_gates(ops)
        assert len(result) == 3

    def test_fused_matrix_correctness(self):
        ops = [Hadamard([0]), Hadamard([0])]
        result = fuse_single_qubit_gates(ops)
        assert len(result) == 1
        expected = np.eye(2, dtype=np.complex128)
        np.testing.assert_allclose(result[0].matrix, expected, atol=1e-10)


class TestFuseAdjacentGates:
    def test_no_fusion_single_op(self):
        ops = [Hadamard([0])]
        result = fuse_adjacent_gates(ops)
        assert len(result) == 1

    def test_fuse_overlapping_gates(self):
        ops = [Hadamard([0]), CX([0, 1]), T([1])]
        result = fuse_adjacent_gates(ops, max_qubits=4)
        assert len(result) == 1
        assert isinstance(result[0], FusedGate)
        assert set(result[0].targets) == {0, 1}

    def test_max_qubits_limit(self):
        ops = [CX([0, 1]), CX([1, 2]), CX([2, 3]), CX([3, 4])]
        result = fuse_adjacent_gates(ops, max_qubits=2)
        assert len(result) == 4

    def test_non_overlapping_not_fused(self):
        ops = [Hadamard([0]), Hadamard([2])]
        result = fuse_adjacent_gates(ops)
        assert len(result) == 2


class TestEstimateFusionBenefit:
    def test_empty_ops(self):
        assert estimate_fusion_benefit([]) == 0.0

    def test_single_op(self):
        assert estimate_fusion_benefit([Hadamard([0])]) == 0.0

    def test_consecutive_single_qubit(self):
        ops = [Hadamard([0]), T([0]), S([0])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0

    def test_no_consecutive(self):
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit == 0.0


class TestFusedGate:
    def test_fused_gate_attributes(self):
        matrix = np.eye(2, dtype=np.complex128)
        gate = FusedGate((0,), matrix)
        assert gate.targets == (0,)
        assert gate.control_state == ()
        assert gate.gate_type == "fused"
        np.testing.assert_array_equal(gate.matrix, matrix)


class TestFuseTwoQubitGates:
    def test_no_fusion_single_op(self):
        ops = [CX([0, 1])]
        result = fuse_two_qubit_gates(ops)
        assert len(result) == 1
        assert result[0] is ops[0]

    def test_fuse_consecutive_same_targets(self):
        ops = [CX([0, 1]), CX([0, 1])]
        result = fuse_two_qubit_gates(ops)
        assert len(result) == 1
        assert isinstance(result[0], FusedGate)
        expected = np.eye(4, dtype=np.complex128)
        np.testing.assert_allclose(result[0].matrix, expected, atol=1e-10)

    def test_fuse_swapped_targets(self):
        ops = [CX([0, 1]), CZ([1, 0])]
        result = fuse_two_qubit_gates(ops)
        assert len(result) == 1
        assert isinstance(result[0], FusedGate)

    def test_no_fusion_different_targets(self):
        ops = [CX([0, 1]), CX([2, 3])]
        result = fuse_two_qubit_gates(ops)
        assert len(result) == 2

    def test_single_qubit_breaks_fusion(self):
        ops = [CX([0, 1]), Hadamard([0]), CX([0, 1])]
        result = fuse_two_qubit_gates(ops)
        assert len(result) == 3


class TestFuseOperations:
    def test_single_qubit_only_mode(self):
        ops = [Hadamard([0]), T([0]), CX([0, 1])]
        result = fuse_operations(ops, single_qubit_only=True)
        assert len(result) == 2
        assert isinstance(result[0], FusedGate)

    def test_full_fusion_mode(self):
        ops = [Hadamard([0]), CX([0, 1]), T([1])]
        result = fuse_operations(ops, max_qubits=4, single_qubit_only=False)
        assert len(result) == 1


class TestOptimizeCircuit:
    def test_single_qubit_fusion_only(self):
        ops = [Hadamard([0]), T([0]), S([0]), CX([0, 1])]
        result = optimize_circuit(
            ops,
            enable_single_qubit_fusion=True,
            enable_two_qubit_fusion=False,
            enable_block_fusion=False,
        )
        assert len(result) == 2
        assert isinstance(result[0], FusedGate)

    def test_two_qubit_fusion_only(self):
        ops = [CX([0, 1]), CX([0, 1]), Hadamard([0])]
        result = optimize_circuit(
            ops,
            enable_single_qubit_fusion=False,
            enable_two_qubit_fusion=True,
            enable_block_fusion=False,
        )
        assert len(result) == 2
        assert isinstance(result[0], FusedGate)

    def test_combined_fusion(self):
        ops = [Hadamard([0]), T([0]), CX([0, 1]), CX([0, 1])]
        result = optimize_circuit(
            ops,
            enable_single_qubit_fusion=True,
            enable_two_qubit_fusion=True,
            enable_block_fusion=False,
        )
        assert len(result) == 2

    def test_short_circuit_no_fusion(self):
        ops = [Hadamard([0])]
        result = optimize_circuit(ops)
        assert len(result) == 1
        assert result[0] is ops[0]


class TestEstimateFusionBenefitTwoQubit:
    def test_consecutive_two_qubit(self):
        ops = [CX([0, 1]), CX([0, 1]), CX([0, 1])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0

    def test_mixed_chains(self):
        ops = [Hadamard([0]), T([0]), CX([0, 1]), CX([0, 1])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0


class TestEstimateFusionBenefitThreeQubit:
    def test_three_qubit_gate_breaks_chains(self):
        from braket.default_simulator.gate_operations import CCNot

        ops = [Hadamard([0]), T([0]), CCNot([0, 1, 2]), S([0])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit >= 0

    def test_three_qubit_gate_resets_counters(self):
        from braket.default_simulator.gate_operations import CCNot

        ops = [CX([0, 1]), CX([0, 1]), CCNot([0, 1, 2]), CX([0, 1]), CX([0, 1])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0

    def test_single_qubit_after_two_qubit_chain(self):
        ops = [CX([0, 1]), CX([0, 1]), Hadamard([0])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0

    def test_three_qubit_after_single_chain(self):
        from braket.default_simulator.gate_operations import CCNot

        ops = [Hadamard([0]), T([0]), S([0]), CCNot([0, 1, 2])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0

    def test_three_qubit_after_two_qubit_chain(self):
        from braket.default_simulator.gate_operations import CCNot

        ops = [CX([0, 1]), CX([0, 1]), CX([0, 1]), CCNot([0, 1, 2])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0

    def test_ends_with_two_qubit_chain(self):
        ops = [Hadamard([0]), CX([0, 1]), CX([0, 1]), CX([0, 1])]
        benefit = estimate_fusion_benefit(ops)
        assert benefit > 0


class TestFuseGateBlocksEdgeCases:
    def test_controlled_gate_not_fused(self):
        from braket.default_simulator.gate_operations import CPhaseShift

        class ControlledOp:
            def __init__(self):
                self.targets = (0,)
                self.matrix = np.eye(2)
                self.control_state = (1,)

        ops = [ControlledOp(), Hadamard([0])]
        result = fuse_adjacent_gates(ops, max_qubits=4)
        assert len(result) == 2

    def test_gate_exceeds_max_qubits(self):
        from braket.default_simulator.gate_operations import CCNot

        ops = [CCNot([0, 1, 2]), Hadamard([0])]
        result = fuse_adjacent_gates(ops, max_qubits=2)
        assert len(result) == 2

    def test_next_op_has_control_state(self):
        class ControlledOp:
            def __init__(self):
                self.targets = (1,)
                self.matrix = np.eye(2)
                self.control_state = (1,)

        ops = [Hadamard([0]), ControlledOp()]
        result = fuse_adjacent_gates(ops, max_qubits=4)
        assert len(result) == 2


class TestEstimateFusionBenefitChainSwitching:
    def test_single_qubit_chain_switches_target(self):
        """Test single-qubit chain followed by different single-qubit target."""
        ops = [Hadamard([0]), T([0]), S([0]), Hadamard([1]), T([1])]
        benefit = estimate_fusion_benefit(ops)
        # First chain on qubit 0 (3 gates), then chain on qubit 1 (2 gates)
        assert benefit > 0

    def test_two_qubit_chain_switches_targets(self):
        """Test two-qubit chain followed by different two-qubit targets."""
        ops = [CX([0, 1]), CX([0, 1]), CX([2, 3]), CX([2, 3])]
        benefit = estimate_fusion_benefit(ops)
        # First chain on [0,1] (2 gates), then chain on [2,3] (2 gates)
        assert benefit > 0


class TestOptimizeCircuitBlockFusion:
    def test_block_fusion_enabled(self):
        """Test optimize_circuit with enable_block_fusion=True."""
        ops = [Hadamard([0]), CX([0, 1]), T([1])]
        result = optimize_circuit(
            ops,
            enable_single_qubit_fusion=False,
            enable_two_qubit_fusion=False,
            enable_block_fusion=True,
        )
        assert len(result) == 1
        assert isinstance(result[0], FusedGate)

    def test_all_fusion_enabled(self):
        """Test optimize_circuit with all fusion types enabled."""
        ops = [Hadamard([0]), T([0]), CX([0, 1]), CX([0, 1]), S([1])]
        result = optimize_circuit(
            ops,
            enable_single_qubit_fusion=True,
            enable_two_qubit_fusion=True,
            enable_block_fusion=True,
        )
        # Should fuse aggressively
        assert len(result) <= len(ops)


class TestAdaptiveOptimize:
    def test_adaptive_optimize_with_benefit(self):
        """Test adaptive_optimize applies fusion when benefit is high."""
        ops = [Hadamard([0]), T([0]), S([0]), CX([0, 1]), CX([0, 1])]
        result = adaptive_optimize(ops)
        # Should fuse since there are chains
        assert len(result) < len(ops)

    def test_adaptive_optimize_no_benefit(self):
        """Test adaptive_optimize skips fusion when benefit is low."""
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2])]
        result = adaptive_optimize(ops)
        # No chains to fuse, should return original
        assert len(result) == len(ops)

    def test_adaptive_optimize_short_circuit(self):
        """Test adaptive_optimize returns early for short circuits."""
        ops = [Hadamard([0])]
        result = adaptive_optimize(ops)
        assert len(result) == 1
        assert result[0] is ops[0]

    def test_adaptive_optimize_max_qubits(self):
        """Test adaptive_optimize respects max_fused_qubits."""
        ops = [Hadamard([0]), T([0]), S([0]), Hadamard([1]), T([1])]
        result = adaptive_optimize(ops, max_fused_qubits=2)
        # Should fuse single-qubit chains
        assert len(result) < len(ops)
