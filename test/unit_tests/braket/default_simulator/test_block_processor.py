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

from braket.default_simulator.block_processor import BlockMatrixProcessor, BlockStructure
from braket.default_simulator.circuit_analyzer import CircuitClass


def hadamard_matrix():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def x_matrix():
    return np.array([[0, 1], [1, 0]], dtype=complex)


class TestBlockStructureIdentification:
    def test_no_block_structure(self):
        processor = BlockMatrixProcessor(4, 2)
        result = processor.identify_block_structure([])
        assert result is None

    def test_simple_controlled_circuit(self):
        processor = BlockMatrixProcessor(4, 2)
        structure = BlockStructure(
            control_qubits=[0, 1],
            target_qubits=[2, 3],
            active_blocks={0},
            block_types={0: CircuitClass.GENERAL},
        )
        assert len(structure.control_qubits) == 2
        assert len(structure.target_qubits) == 2

    def test_multi_control_structure(self):
        processor = BlockMatrixProcessor(5, 3)
        assert processor.n_blocks == 8
        assert processor.block_size == 4


class TestBlockInitialization:
    def test_uniform_block_types(self):
        processor = BlockMatrixProcessor(4, 2)
        block_types = {i: CircuitClass.GENERAL for i in range(4)}
        processor.initialize_blocks(block_types)
        assert len(processor.blocks) == 4

    def test_mixed_block_types(self):
        processor = BlockMatrixProcessor(4, 2)
        block_types = {
            0: CircuitClass.PRODUCT,
            1: CircuitClass.CLIFFORD,
            2: CircuitClass.GENERAL,
            3: CircuitClass.DIAGONAL,
        }
        processor.initialize_blocks(block_types)
        assert len(processor.blocks) == 4

    def test_sparse_active_blocks(self):
        processor = BlockMatrixProcessor(4, 2)
        block_types = {0: CircuitClass.GENERAL, 2: CircuitClass.GENERAL}
        processor.initialize_blocks(block_types)
        assert 0 in processor.blocks
        assert 2 in processor.blocks


class TestBlockLocalOperations:
    def test_single_block_operation(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})

        identity = np.eye(4, dtype=complex)
        processor.apply_block_local_operation(0, identity)

        state = processor.blocks[0]
        assert np.isclose(state[0], 1.0, atol=1e-7)

    def test_all_blocks_operation(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})

        identity = np.eye(4, dtype=complex)
        for block_idx in processor.blocks:
            processor.apply_block_local_operation(block_idx, identity)

    def test_block_independence(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})

        x_4 = np.kron(x_matrix(), np.eye(2))
        processor.apply_block_local_operation(0, x_4)

        assert not np.allclose(processor.blocks[0], processor.blocks[1])


class TestBlockDiagonalOperations:
    def test_uniform_diagonal(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})

        identity = np.eye(4, dtype=complex)
        block_unitaries = {0: identity, 1: identity}
        processor.apply_block_diagonal_unitary(block_unitaries)

    def test_varied_diagonal(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})

        x_4 = np.kron(x_matrix(), np.eye(2))
        block_unitaries = {0: np.eye(4, dtype=complex), 1: x_4}
        processor.apply_block_diagonal_unitary(block_unitaries)


class TestBlockMixingOperations:
    def test_hadamard_mixing(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0}

        h = hadamard_matrix()
        processor.apply_block_mixing_operation(h)

        assert len(processor.block_amplitudes) == 2
        assert np.isclose(np.abs(processor.block_amplitudes[0]), 1 / np.sqrt(2), atol=1e-7)
        assert np.isclose(np.abs(processor.block_amplitudes[1]), 1 / np.sqrt(2), atol=1e-7)

    def test_arbitrary_mixing(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0, 1: 0.0}

        mixing = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=complex)
        processor.apply_block_mixing_operation(mixing)

        assert np.isclose(np.abs(processor.block_amplitudes[0]), 0.6, atol=1e-7)
        assert np.isclose(np.abs(processor.block_amplitudes[1]), 0.8, atol=1e-7)

    def test_block_count_change(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0}

        initial_count = len(processor.get_active_blocks())
        h = hadamard_matrix()
        processor.apply_block_mixing_operation(h)
        final_count = len(processor.get_active_blocks())

        assert final_count >= initial_count


class TestBlockAmplitudes:
    def test_single_active_block(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0}

        amp = processor.get_amplitude(0)
        assert np.isclose(amp, 1.0, atol=1e-7)

        amp = processor.get_amplitude(4)
        assert np.isclose(amp, 0.0, atol=1e-7)

    def test_multiple_active_blocks(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1 / np.sqrt(2), 1: 1 / np.sqrt(2)}

        amp_0 = processor.get_amplitude(0)
        amp_4 = processor.get_amplitude(4)
        assert np.isclose(np.abs(amp_0), 1 / np.sqrt(2), atol=1e-7)
        assert np.isclose(np.abs(amp_4), 1 / np.sqrt(2), atol=1e-7)

    def test_amplitude_normalization(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1 / np.sqrt(2), 1: 1 / np.sqrt(2)}

        total_prob = 0
        for i in range(8):
            amp = processor.get_amplitude(i)
            total_prob += np.abs(amp) ** 2

        assert np.isclose(total_prob, 1.0, atol=1e-7)


class TestBlockSampling:
    def test_single_block_sampling(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0}

        results = processor.sample(100)
        for bitstring in results:
            assert bitstring[0] == "0"

    def test_multi_block_sampling(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1 / np.sqrt(2), 1: 1 / np.sqrt(2)}

        results = processor.sample(1000)
        block_0_count = sum(v for k, v in results.items() if k[0] == "0")
        block_1_count = sum(v for k, v in results.items() if k[0] == "1")

        assert 300 < block_0_count < 700
        assert 300 < block_1_count < 700

    def test_block_probability_weighting(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: np.sqrt(0.9), 1: np.sqrt(0.1)}

        results = processor.sample(10000)
        block_0_count = sum(v for k, v in results.items() if k[0] == "0")

        assert block_0_count > 8000


class TestBlockPruning:
    def test_prune_zero_blocks(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0, 1: 0.0}

        processor.prune_negligible_blocks(threshold=1e-10)
        assert 1 not in processor.block_amplitudes

    def test_prune_threshold(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0, 1: 1e-12}

        processor.prune_negligible_blocks(threshold=1e-10)
        assert 1 not in processor.block_amplitudes

    def test_prune_preserves_normalization(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 0.99, 1: 0.01}

        processor.prune_negligible_blocks(threshold=0.05)

        total_prob = sum(
            np.abs(amp) ** 2 * np.sum(np.abs(processor.blocks.get(idx, [0])) ** 2)
            for idx, amp in processor.block_amplitudes.items()
        )
        assert np.isclose(total_prob, 1.0, atol=1e-7)


class TestBlockStructureWithOperations:
    def test_identify_with_single_qubit_ops(self):
        from braket.default_simulator.gate_operations import Hadamard

        processor = BlockMatrixProcessor(4, 2)
        ops = [Hadamard([2]), Hadamard([3])]
        structure = processor.identify_block_structure(ops)
        assert structure is not None
        assert structure.control_qubits == [0, 1]
        assert structure.target_qubits == [2, 3]

    def test_identify_with_cross_partition_ops(self):
        from braket.default_simulator.gate_operations import CX

        processor = BlockMatrixProcessor(4, 2)
        ops = [CX([1, 2])]
        structure = processor.identify_block_structure(ops)
        assert structure is None


class TestBlockStateManagement:
    def test_set_block_state(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        new_state = np.array([0, 1, 0, 0], dtype=complex)
        processor.set_block_state(0, new_state)
        assert np.allclose(processor.blocks[0], new_state)

    def test_set_block_state_invalid_size(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        with pytest.raises(ValueError, match="doesn't match block size"):
            processor.set_block_state(0, np.array([1, 0]))

    def test_set_block_amplitude(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.set_block_amplitude(0, 0.5 + 0.5j)
        assert processor.block_amplitudes[0] == 0.5 + 0.5j


class TestBlockLocalOperationErrors:
    def test_apply_to_nonexistent_block(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.apply_block_local_operation(5, np.eye(4))

    def test_apply_wrong_size_matrix(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        with pytest.raises(ValueError, match="doesn't match block size"):
            processor.apply_block_local_operation(0, np.eye(2))


class TestBlockMixingErrors:
    def test_mixing_wrong_size(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        with pytest.raises(ValueError, match="doesn't match number of blocks"):
            processor.apply_block_mixing_operation(np.eye(4))


class TestBlockStateVector:
    def test_get_state_vector(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL, 1: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1 / np.sqrt(2), 1: 1 / np.sqrt(2)}
        state = processor.get_state_vector()
        assert len(state) == 8
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_get_state_vector_single_block(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0}
        state = processor.get_state_vector()
        assert np.isclose(state[0], 1.0, atol=1e-7)
        assert np.isclose(np.sum(np.abs(state[4:])), 0.0, atol=1e-7)


class TestBlockProcessorEdgeCases:
    def test_get_amplitude_missing_block(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        processor.block_amplitudes = {0: 1.0}
        amp = processor.get_amplitude(7)
        assert amp == 0.0j

    def test_prune_empty_amplitudes(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {}
        processor.block_amplitudes = {}
        processor.prune_negligible_blocks()
        assert len(processor.block_amplitudes) == 0

    def test_apply_diagonal_to_missing_block(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.initialize_blocks({0: CircuitClass.GENERAL})
        block_unitaries = {0: np.eye(4), 5: np.eye(4)}
        processor.apply_block_diagonal_unitary(block_unitaries)
        assert 0 in processor.blocks
        assert 5 not in processor.blocks

    def test_mixing_creates_new_blocks(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {0: np.array([1, 0, 0, 0], dtype=complex)}
        processor.block_amplitudes = {0: 1.0}
        h = hadamard_matrix()
        processor.apply_block_mixing_operation(h)
        assert 1 in processor.blocks

    def test_get_state_vector_missing_block_in_amplitudes(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {0: np.array([1, 0, 0, 0], dtype=complex)}
        processor.block_amplitudes = {0: 1.0, 1: 0.5}
        state = processor.get_state_vector()
        assert len(state) == 8


class TestBlockProcessorBranchCoverage:
    def test_identify_structure_single_qubit_two_qubit_mix(self):
        from braket.default_simulator.gate_operations import CX, Hadamard

        processor = BlockMatrixProcessor(4, 2)
        ops = [Hadamard([2]), CX([2, 3])]
        structure = processor.identify_block_structure(ops)
        assert structure is not None

    def test_mixing_with_zero_amplitude_result(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {0: np.array([1, 0, 0, 0], dtype=complex)}
        processor.block_amplitudes = {0: 1.0}
        zero_mixing = np.array([[0, 1], [1, 0]], dtype=complex)
        processor.apply_block_mixing_operation(zero_mixing)
        assert 0 not in processor.block_amplitudes or processor.block_amplitudes.get(0, 0) == 0

    def test_sample_with_non_normalized_state(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {0: np.array([0.5, 0.5, 0, 0], dtype=complex)}
        processor.block_amplitudes = {0: 1.0}
        results = processor.sample(100)
        total = sum(results.values())
        assert total == 100

    def test_get_active_blocks_with_negligible_amplitudes(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.block_amplitudes = {0: 1.0, 1: 1e-20}
        active = processor.get_active_blocks()
        assert 0 in active
        assert 1 not in active

    def test_mixing_produces_negligible_amplitude(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {0: np.array([1, 0, 0, 0], dtype=complex)}
        processor.block_amplitudes = {0: 1e-20}
        identity = np.eye(2, dtype=complex)
        processor.apply_block_mixing_operation(identity)
        assert len(processor.block_amplitudes) == 0

    def test_mixing_block_already_exists(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {
            0: np.array([1, 0, 0, 0], dtype=complex),
            1: np.array([0, 1, 0, 0], dtype=complex),
        }
        processor.block_amplitudes = {0: 1.0}
        h = hadamard_matrix()
        processor.apply_block_mixing_operation(h)
        assert np.allclose(processor.blocks[1], [0, 1, 0, 0])

    def test_prune_with_zero_total_prob(self):
        processor = BlockMatrixProcessor(3, 1)
        processor.blocks = {}
        processor.block_amplitudes = {0: 1e-15}
        processor.prune_negligible_blocks(threshold=1e-10)
        assert len(processor.block_amplitudes) == 0
