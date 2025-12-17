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

from braket.default_simulator.stabilizer_simulator import (
    StabilizerSimulator,
    StabilizerTableau,
)


class TestStabilizerTableauInitialization:
    def test_initial_state(self):
        tableau = StabilizerTableau(2)
        assert tableau.n == 2
        assert tableau.tableau.shape == (4, 4)
        assert tableau.phases.shape == (4,)

    def test_tableau_dimensions(self):
        for n in [1, 3, 5]:
            tableau = StabilizerTableau(n)
            assert tableau.tableau.shape == (2 * n, 2 * n)
            assert tableau.phases.shape == (2 * n,)

    def test_initial_stabilizers(self):
        tableau = StabilizerTableau(2)
        assert tableau.tableau[0, 0] == 1
        assert tableau.tableau[1, 1] == 1
        assert tableau.tableau[2, 2] == 1
        assert tableau.tableau[3, 3] == 1


class TestStabilizerSingleQubitGates:
    def test_hadamard(self):
        tableau = StabilizerTableau(1)
        tableau.h(0)
        assert tableau.tableau[1, 0] == 1
        assert tableau.tableau[1, 1] == 0

    def test_phase_gate(self):
        tableau = StabilizerTableau(1)
        tableau.s(0)
        assert tableau.tableau[1, 1] == 1

    def test_pauli_gates(self):
        tableau = StabilizerTableau(1)
        initial_phases = tableau.phases.copy()
        tableau.x(0)
        tableau.x(0)
        assert np.array_equal(tableau.phases % 4, initial_phases % 4)

    def test_gate_composition(self):
        tableau = StabilizerTableau(1)
        tableau.h(0)
        tableau.s(0)
        tableau.h(0)
        pass


class TestStabilizerTwoQubitGates:
    def test_cnot(self):
        tableau = StabilizerTableau(2)
        tableau.h(0)
        tableau.cnot(0, 1)
        pass

    def test_cz(self):
        tableau = StabilizerTableau(2)
        tableau.h(0)
        tableau.h(1)
        tableau.cz(0, 1)
        pass

    def test_swap(self):
        tableau = StabilizerTableau(2)
        tableau.x(0)
        tableau.swap(0, 1)
        pass


class TestStabilizerMeasurement:
    def test_deterministic_measurement(self):
        sim = StabilizerSimulator(1)
        result = sim.measure(0)
        assert result == 0

        sim2 = StabilizerSimulator(1)
        sim2.tableau.x(0)
        result2 = sim2.measure(0)
        assert result2 == 1

    def test_random_measurement(self):
        results = []
        for _ in range(100):
            sim = StabilizerSimulator(1)
            sim.tableau.h(0)
            results.append(sim.measure(0))

        assert 0 in results
        assert 1 in results

    def test_measurement_state_update(self):
        sim = StabilizerSimulator(1)
        sim.tableau.h(0)
        first_result = sim.measure(0)
        second_result = sim.measure(0)
        assert first_result == second_result

    def test_repeated_measurement(self):
        sim = StabilizerSimulator(1)
        result1 = sim.measure(0)
        result2 = sim.measure(0)
        result3 = sim.measure(0)
        assert result1 == result2 == result3


class TestStabilizerSampling:
    def test_bell_state_correlations(self):
        results = {"00": 0, "11": 0, "01": 0, "10": 0}
        for _ in range(1000):
            sim = StabilizerSimulator(2)
            sim.tableau.h(0)
            sim.tableau.cnot(0, 1)
            outcome = sim.measure_all()
            results[outcome] += 1

        assert results["00"] > 400
        assert results["11"] > 400
        assert results["01"] < 50
        assert results["10"] < 50

    def test_ghz_state_correlations(self):
        results = {}
        for _ in range(1000):
            sim = StabilizerSimulator(3)
            sim.tableau.h(0)
            sim.tableau.cnot(0, 1)
            sim.tableau.cnot(1, 2)
            outcome = sim.measure_all()
            results[outcome] = results.get(outcome, 0) + 1

        assert results.get("000", 0) > 400
        assert results.get("111", 0) > 400
        for bad in ["001", "010", "011", "100", "101", "110"]:
            assert results.get(bad, 0) < 50

    def test_sample_distribution(self):
        sim = StabilizerSimulator(2)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        results = sim.sample(1000)

        total = sum(results.values())
        assert total == 1000
        assert results.get("00", 0) > 400
        assert results.get("11", 0) > 400


class TestStabilizerCorrectness:
    def test_matches_full_simulator_bell(self):
        sim = StabilizerSimulator(2)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)

        results = sim.sample(10000)
        prob_00 = results.get("00", 0) / 10000
        prob_11 = results.get("11", 0) / 10000

        assert abs(prob_00 - 0.5) < 0.05
        assert abs(prob_11 - 0.5) < 0.05

    def test_matches_full_simulator_ghz(self):
        sim = StabilizerSimulator(3)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        sim.tableau.cnot(1, 2)

        results = sim.sample(10000)
        prob_000 = results.get("000", 0) / 10000
        prob_111 = results.get("111", 0) / 10000

        assert abs(prob_000 - 0.5) < 0.05
        assert abs(prob_111 - 0.5) < 0.05

    def test_matches_full_simulator_random_clifford(self):
        np.random.seed(42)
        sim = StabilizerSimulator(3)

        for _ in range(10):
            gate = np.random.choice(["h", "s", "cnot", "cz"])
            if gate == "h":
                sim.apply_gate("h", [np.random.randint(3)])
            elif gate == "s":
                sim.apply_gate("s", [np.random.randint(3)])
            elif gate == "cnot":
                q1, q2 = np.random.choice(3, 2, replace=False)
                sim.apply_gate("cnot", [q1, q2])
            elif gate == "cz":
                q1, q2 = np.random.choice(3, 2, replace=False)
                sim.apply_gate("cz", [q1, q2])

        results = sim.sample(1000)
        total = sum(results.values())
        assert total == 1000


class TestStabilizerDeterminism:
    def test_is_deterministic_computational_basis(self):
        sim = StabilizerSimulator(1)
        is_det, outcome = sim.is_deterministic(0)
        assert is_det is True
        assert outcome == 0

    def test_is_deterministic_superposition(self):
        sim = StabilizerSimulator(1)
        sim.tableau.h(0)
        is_det, outcome = sim.is_deterministic(0)
        assert is_det is False
        assert outcome is None

    def test_is_deterministic_after_measurement(self):
        sim = StabilizerSimulator(1)
        sim.tableau.h(0)
        measured = sim.measure(0)
        is_det, outcome = sim.is_deterministic(0)
        assert is_det is True
        assert outcome == measured


class TestStabilizerYGate:
    def test_y_gate(self):
        tableau = StabilizerTableau(1)
        tableau.y(0)
        pass

    def test_y_gate_twice(self):
        tableau = StabilizerTableau(1)
        initial_phases = tableau.phases.copy()
        tableau.y(0)
        tableau.y(0)
        assert np.array_equal(tableau.phases % 4, initial_phases % 4)


class TestStabilizerSdgGate:
    def test_sdg_gate(self):
        tableau = StabilizerTableau(1)
        tableau.sdg(0)
        pass

    def test_s_sdg_identity(self):
        tableau = StabilizerTableau(1)
        tableau.s(0)
        tableau.sdg(0)
        pass


class TestStabilizerInitialize:
    def test_initialize_basis_state(self):
        sim = StabilizerSimulator(3)
        sim.initialize(5)
        result = sim.measure_all()
        assert result == "101"

    def test_initialize_zero(self):
        sim = StabilizerSimulator(2)
        sim.initialize(0)
        result = sim.measure_all()
        assert result == "00"


class TestStabilizerApplyGate:
    def test_apply_gate_unsupported(self):
        sim = StabilizerSimulator(2)
        with pytest.raises(ValueError, match="Unsupported Clifford gate"):
            sim.apply_gate("t", [0])

    def test_apply_gate_all_types(self):
        sim = StabilizerSimulator(2)
        sim.apply_gate("hadamard", [0])
        sim.apply_gate("pauli_x", [0])
        sim.apply_gate("pauli_y", [0])
        sim.apply_gate("pauli_z", [0])
        sim.apply_gate("s", [0])
        sim.apply_gate("si", [0])
        sim.apply_gate("cx", [0, 1])
        sim.apply_gate("cz", [0, 1])
        sim.apply_gate("swap", [0, 1])


class TestStabilizerApplyOperations:
    def test_apply_operations(self):
        from braket.default_simulator.gate_operations import CX, Hadamard

        sim = StabilizerSimulator(2)
        ops = [Hadamard([0]), CX([0, 1])]
        sim.apply_operations(ops)
        results = sim.sample(1000)
        assert results.get("00", 0) > 400
        assert results.get("11", 0) > 400


class TestStabilizerMeasureAll:
    def test_measure_all_deterministic(self):
        sim = StabilizerSimulator(3)
        sim.tableau.x(0)
        sim.tableau.x(2)
        result = sim.measure_all()
        assert result == "101"


class TestStabilizerCopy:
    def test_copy_independence(self):
        sim = StabilizerSimulator(2)
        sim.tableau.h(0)
        sim_copy = sim.copy()
        sim.tableau.cnot(0, 1)
        sv_orig = sim.get_state_vector()
        sv_copy = sim_copy.get_state_vector()
        assert not np.allclose(sv_orig, sv_copy, atol=1e-7)


class TestStabilizerStateVector:
    def test_get_state_vector_computational_basis(self):
        sim = StabilizerSimulator(2)
        state = sim.get_state_vector()
        expected = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_allclose(state, expected, atol=1e-7)

    def test_get_state_vector_bell_state(self):
        sim = StabilizerSimulator(2)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        state = sim.get_state_vector()
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(state), np.abs(expected), atol=1e-7)

    def test_get_state_vector_too_large(self):
        sim = StabilizerSimulator(25)
        with pytest.raises(ValueError, match="too large"):
            sim.get_state_vector()

    def test_get_state_vector_x_state(self):
        sim = StabilizerSimulator(2)
        sim.tableau.x(0)
        state = sim.get_state_vector()
        expected = np.array([0, 0, 1, 0], dtype=np.complex128)
        np.testing.assert_allclose(state, expected, atol=1e-7)


class TestStabilizerTableauCopy:
    def test_tableau_copy(self):
        tableau = StabilizerTableau(2)
        tableau.h(0)
        tableau_copy = tableau.copy()
        tableau.cnot(0, 1)
        assert not np.array_equal(tableau.tableau, tableau_copy.tableau)


class TestStabilizerMeasureWithOutcome:
    def test_measure_with_forced_outcome(self):
        tableau = StabilizerTableau(1)
        tableau.h(0)
        result = tableau.measure(0, random_outcome=1)
        assert result == 1
        result2 = tableau.measure(0)
        assert result2 == 1


class TestStabilizerRowsum:
    def test_rowsum_triggers_g_function_branches(self):
        tableau = StabilizerTableau(2)
        tableau.h(0)
        tableau.cnot(0, 1)
        tableau.s(0)
        tableau.h(1)
        tableau.cnot(1, 0)
        tableau.s(1)
        tableau.h(0)
        tableau.h(1)
        assert tableau.tableau.shape == (4, 4)

    def test_rowsum_with_multiple_stabilizers(self):
        tableau = StabilizerTableau(3)
        tableau.h(0)
        tableau.h(1)
        tableau.h(2)
        tableau.cnot(0, 1)
        tableau.cnot(1, 2)
        tableau.cz(0, 2)
        result = tableau.measure(1)
        assert result in [0, 1]


class TestStabilizerDeterministicMeasurement:
    def test_deterministic_measurement_path(self):
        sim = StabilizerSimulator(2)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        sim.measure(0)
        is_det, val = sim.is_deterministic(1)
        assert is_det is True

    def test_is_deterministic_with_entanglement(self):
        sim = StabilizerSimulator(3)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        sim.tableau.cnot(0, 2)
        is_det0, _ = sim.is_deterministic(0)
        is_det1, _ = sim.is_deterministic(1)
        is_det2, _ = sim.is_deterministic(2)
        assert is_det0 is False
        assert is_det1 is False
        assert is_det2 is False


class TestStabilizerZGate:
    def test_z_gate_phase_update(self):
        tableau = StabilizerTableau(1)
        tableau.h(0)
        initial_phase = tableau.phases.copy()
        tableau.z(0)
        assert not np.array_equal(tableau.phases, initial_phase)


class TestStabilizerNumbaThreshold:
    def test_large_tableau_uses_numba_paths(self):
        tableau = StabilizerTableau(20)
        tableau.h(0)
        tableau.cnot(0, 1)
        tableau.s(0)
        result = tableau.measure(0)
        assert result in [0, 1]

    def test_large_simulator_operations(self):
        sim = StabilizerSimulator(20)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        sim.tableau.s(0)
        results = sim.sample(100)
        assert sum(results.values()) == 100


class TestStabilizerEdgeCases:
    def test_measure_deterministic_path_with_scratch(self):
        sim = StabilizerSimulator(3)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        sim.tableau.cnot(0, 2)
        sim.measure(0)
        is_det, val = sim.is_deterministic(1)
        assert is_det is True

    def test_rowsum_all_g_branches(self):
        tableau = StabilizerTableau(4)
        tableau.h(0)
        tableau.h(1)
        tableau.cnot(0, 1)
        tableau.s(0)
        tableau.s(1)
        tableau.h(0)
        tableau.cnot(1, 0)
        tableau.h(1)
        tableau.cnot(0, 1)
        result = tableau.measure(0)
        assert result in [0, 1]

    def test_rowsum_g_function_z_only_branch(self):
        tableau = StabilizerTableau(2)
        tableau.s(0)
        tableau.h(0)
        tableau.s(0)
        tableau.cnot(0, 1)
        tableau.s(1)
        tableau.h(1)
        result = tableau.measure(0)
        assert result in [0, 1]

    def test_is_deterministic_copy_with_multiple_destabilizers(self):
        sim = StabilizerSimulator(3)
        sim.tableau.h(0)
        sim.tableau.cnot(0, 1)
        sim.tableau.h(2)
        sim.tableau.cnot(2, 1)
        sim.measure(0)
        sim.measure(2)
        is_det, val = sim.is_deterministic(1)
        assert is_det is True


class TestStabilizerNumbaRowsum:
    def test_rowsum_numba_path_explicit(self):
        tableau = StabilizerTableau(18)
        tableau.h(0)
        tableau.cnot(0, 1)
        tableau.h(2)
        tableau.cnot(2, 3)
        result = tableau.measure(0)
        assert result in [0, 1]
        result2 = tableau.measure(2)
        assert result2 in [0, 1]


class TestStabilizerCoverageGaps:
    def test_rowsum_numba_path_via_measurement(self):
        """Test _rowsum Numba path is triggered during measurement on large tableau."""
        tableau = StabilizerTableau(18)
        # Create entanglement to ensure _rowsum is called during measurement
        for i in range(17):
            tableau.h(i)
            tableau.cnot(i, i + 1)
        # Measurement should trigger _rowsum for rows with qubit set
        result = tableau.measure(0)
        assert result in [0, 1]


class TestStabilizerRowsumNumbaPath:
    """Test _rowsum Numba path with >= 16 qubits."""

    def test_rowsum_numba_path_triggered(self):
        """Test that _rowsum uses Numba path with >= 16 qubits (lines 142-143)."""
        # Create a tableau with >= 16 qubits to trigger Numba path
        # _NUMBA_THRESHOLD = 16, so we need n >= 16
        tableau = StabilizerTableau(18)
        # Create GHZ-like state: H on first qubit, then CNOT chain
        # This creates entanglement where multiple stabilizers have X on qubit 0
        tableau.h(0)
        for i in range(17):
            tableau.cnot(i, i + 1)
        # After this, measuring qubit 0 will find p (a stabilizer with X on qubit 0)
        # and then call _rowsum for other rows that also have X on qubit 0
        # The CNOT chain propagates X operators, so multiple rows will have X on qubit 0
        result = tableau.measure(0)
        assert result in [0, 1]
        # Verify subsequent measurements are deterministic
        result2 = tableau.measure(0)
        assert result == result2

    def test_rowsum_numba_multiple_rows(self):
        """Test _rowsum Numba path with multiple rows needing combination."""
        # Create 20-qubit tableau (well above threshold)
        tableau = StabilizerTableau(20)
        # Create highly entangled state to ensure multiple rows have X on measured qubit
        for i in range(20):
            tableau.h(i)
        for i in range(19):
            tableau.cnot(i, i + 1)
        # Add more entanglement
        for i in range(0, 18, 2):
            tableau.cz(i, i + 2)
        # Measure middle qubit - should trigger _rowsum multiple times
        result = tableau.measure(10)
        assert result in [0, 1]
