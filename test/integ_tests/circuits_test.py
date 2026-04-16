"""Braket LocalSimulator validation tests.

Compares shot-based results from LocalSimulator against a numpy reference
statevector simulator for various circuit patterns.
"""

import numpy as np
import pytest
from braket.circuits import Circuit
from braket.default_simulator.gate_operations import (
    CCNot,
    CSwap as CSwapGate,
    CX,
    CZ,
    Hadamard,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    S,
    Swap,
    T,
)
from braket.devices import LocalSimulator

GATE_MATRICES = {
    "h": Hadamard._matrix,
    "x": PauliX._matrix,
    "y": PauliY._matrix,
    "z": PauliZ._matrix,
    "s": S._matrix,
    "t": T._matrix,
    "i": Identity._matrix,
    "cnot": CX._matrix,
    "cz": CZ._matrix,
    "swap": Swap._matrix,
    "ccnot": CCNot._matrix,
    "cswap": CSwapGate._matrix,
}

ONE_QUBIT_GATES = ["h", "x", "y", "z", "s", "t", "i"]
TWO_QUBIT_GATES = ["cnot", "cz", "swap"]
THREE_QUBIT_GATES = ["ccnot", "cswap"]

SHOTS = 2000
ATOL = 0.04


def _apply_gate(state, n_qubits, matrix, targets):
    n_gate = len(targets)
    dim = 2**n_qubits
    state = state.reshape([2] * n_qubits)
    mat = matrix.reshape([2] * (2 * n_gate))
    source_axes = list(range(n_gate, 2 * n_gate))
    state = np.tensordot(mat, state, axes=(source_axes, targets))
    current_order = list(targets) + [i for i in range(n_qubits) if i not in targets]
    inv = [0] * n_qubits
    for i, v in enumerate(current_order):
        inv[v] = i
    state = state.transpose(inv)
    return state.reshape(dim)


def _reference_probabilities(ops, n_qubits):
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    for gate_name, targets in ops:
        state = _apply_gate(state, n_qubits, GATE_MATRICES[gate_name], targets)
    return np.abs(state) ** 2


def _to_braket(ops, n_qubits):
    c = Circuit()
    for gate_name, targets in ops:
        getattr(c, gate_name)(*targets)
    for q in range(n_qubits):
        c.measure(q)
    return c


def _measured_probabilities(ops, n_qubits):
    circuit = _to_braket(ops, n_qubits)
    result = LocalSimulator().run(circuit, shots=SHOTS).result()
    measured = np.zeros(2**n_qubits)
    qubit_order = result.measured_qubits
    for shot in result.measurements:
        idx = 0
        for i in range(n_qubits):
            col = qubit_order.index(i)
            idx |= int(shot[col]) << (n_qubits - 1 - i)
        measured[idx] += 1
    measured /= SHOTS
    return measured


def _assert_distributions_close(ops, n_qubits):
    expected = _reference_probabilities(ops, n_qubits)
    measured = _measured_probabilities(ops, n_qubits)
    max_diff = np.max(np.abs(expected - measured))
    if max_diff > ATOL:
        worst = np.argmax(np.abs(expected - measured))
        pytest.fail(
            f"|diff|={max_diff:.4f} at |{worst:0{n_qubits}b}> "
            f"(expected={expected[worst]:.4f}, observed={measured[worst]:.4f})"
        )


def _assert_exact_probabilities(ops, n_qubits, expected_probs):
    actual = _reference_probabilities(ops, n_qubits)
    np.testing.assert_allclose(actual, expected_probs, atol=1e-12)


def _assert_reference_matches_braket_exact(ops, n_qubits):
    expected = _reference_probabilities(ops, n_qubits)
    c = Circuit()
    for gate_name, targets in ops:
        getattr(c, gate_name)(*targets)
    for q in range(n_qubits):
        c.i(q)
    c.probability()
    result = LocalSimulator().run(c, shots=0).result()
    measured = np.array(result.result_types[0].value)
    np.testing.assert_allclose(measured, expected, atol=1e-10)



def _random_layered_circuit(rng, n_qubits, n_layers, gate_set=None):
    gate_set = gate_set or ONE_QUBIT_GATES + TWO_QUBIT_GATES
    ops = []
    for _ in range(n_layers):
        available = list(range(n_qubits))
        rng.shuffle(available)
        while available:
            g = rng.choice(gate_set)
            if g in ONE_QUBIT_GATES and len(available) >= 1:
                ops.append((g, [available.pop()]))
            elif g in TWO_QUBIT_GATES and len(available) >= 2:
                ops.append((g, [available.pop(), available.pop()]))
            elif g in THREE_QUBIT_GATES and len(available) >= 3:
                ops.append((g, [available.pop(), available.pop(), available.pop()]))
            else:
                break
    return ops


def _ghz_circuit(n_qubits, ghz_start, n_ghz, use_tree):
    ops = []
    ghz_qs = list(range(ghz_start, ghz_start + n_ghz))
    for q in range(n_qubits):
        if q not in ghz_qs:
            ops.append(("i", [q]))
    ops.append(("h", [ghz_qs[0]]))
    if use_tree:
        layer = [ghz_qs[0]]
        idx = 1
        while idx < len(ghz_qs):
            next_layer = []
            for ctrl in layer:
                if idx < len(ghz_qs):
                    ops.append(("cnot", [ctrl, ghz_qs[idx]]))
                    next_layer.extend([ctrl, ghz_qs[idx]])
                    idx += 1
            layer = next_layer
    else:
        for i in range(1, len(ghz_qs)):
            ops.append(("cnot", [ghz_qs[i - 1], ghz_qs[i]]))
    return ops


def _sparse_qubit_circuit(rng, qubit_indices, n_layers):
    ops = []
    for _ in range(n_layers):
        idxs = list(qubit_indices)
        rng.shuffle(idxs)
        while len(idxs) >= 2:
            g = rng.choice(ONE_QUBIT_GATES + TWO_QUBIT_GATES)
            if g in ONE_QUBIT_GATES:
                ops.append((g, [idxs.pop()]))
            elif len(idxs) >= 2:
                ops.append((g, [idxs.pop(), idxs.pop()]))
    return ops, max(qubit_indices) + 1


def _bell_pair(q0, q1, n_qubits):
    ops = [("h", [q0]), ("cnot", [q0, q1])]
    for q in range(n_qubits):
        if q not in (q0, q1):
            ops.append(("i", [q]))
    return ops


def _qft_circuit(n_qubits, targets=None):
    targets = targets or list(range(n_qubits))
    ops = []
    for i, qi in enumerate(targets):
        ops.append(("h", [qi]))
        for j in range(i + 1, len(targets)):
            ops.append(("cz", [targets[j], qi]))
    for i in range(len(targets) // 2):
        ops.append(("swap", [targets[i], targets[len(targets) - 1 - i]]))
    return ops


def _grover_oracle_diffuser(n_qubits, marked_state):
    ops = []
    for q in range(n_qubits):
        ops.append(("h", [q]))

    for q in range(n_qubits):
        if not (marked_state >> (n_qubits - 1 - q)) & 1:
            ops.append(("x", [q]))

    if n_qubits == 3:
        ops.append(("ccnot", [0, 1, 2]))
    else:
        ops.append(("cz", [0, 1]))

    for q in range(n_qubits):
        if not (marked_state >> (n_qubits - 1 - q)) & 1:
            ops.append(("x", [q]))

    for q in range(n_qubits):
        ops.append(("h", [q]))
        ops.append(("x", [q]))

    if n_qubits == 3:
        ops.append(("ccnot", [0, 1, 2]))
    else:
        ops.append(("cz", [0, 1]))

    for q in range(n_qubits):
        ops.append(("x", [q]))
        ops.append(("h", [q]))

    return ops


def _swap_network(n_qubits, rounds=1):
    ops = []
    for _ in range(rounds):
        for start in range(2):
            for i in range(start, n_qubits - 1, 2):
                ops.append(("swap", [i, i + 1]))
    return ops


def _repeated_gate_circuit(gate, n_qubits, repetitions):
    ops = []
    for _ in range(repetitions):
        for q in range(n_qubits):
            ops.append((gate, [q]))
    return ops


def _reverse_cnot_ladder(n_qubits):
    ops = [("h", [q]) for q in range(n_qubits)]
    for i in range(n_qubits - 1, 0, -1):
        ops.append(("cnot", [i, i - 1]))
    return ops


def _w_state_approx(n_qubits):
    ops = [("x", [0])]
    for i in range(n_qubits - 1):
        ops.append(("h", [i]))
        ops.append(("cnot", [i, i + 1]))
    return ops


def _checkerboard_circuit(n_qubits, n_layers):
    ops = []
    for layer in range(n_layers):
        for q in range(n_qubits):
            ops.append(("h" if (q + layer) % 2 == 0 else "t", [q]))
        start = layer % 2
        for i in range(start, n_qubits - 1, 2):
            ops.append(("cnot", [i, i + 1]))
    return ops


def _long_range_cnot_circuit(n_qubits, stride):
    ops = [("h", [q]) for q in range(n_qubits)]
    for i in range(n_qubits):
        target = (i + stride) % n_qubits
        if target != i:
            ops.append(("cnot", [i, target]))
    return ops


def _cascade_circuit(n_qubits):
    ops = [("h", [0])]
    for i in range(n_qubits - 1):
        ops.append(("cnot", [i, i + 1]))
        ops.append(("s", [i + 1]))
    return ops


def _identity_sandwich(n_qubits, inner_ops):
    ops = []
    for q in range(n_qubits):
        ops.append(("i", [q]))
    ops.extend(inner_ops)
    for q in range(n_qubits):
        ops.append(("i", [q]))
    return ops


def _alternating_basis_circuit(n_qubits, n_layers):
    ops = []
    for layer in range(n_layers):
        for q in range(n_qubits):
            ops.append(("h", [q]))
        for i in range(0, n_qubits - 1, 2):
            ops.append(("cz", [i, i + 1]))
        for q in range(n_qubits):
            ops.append(("s", [q]))
    return ops


def _dense_entangle_circuit(n_qubits):
    ops = [("h", [q]) for q in range(n_qubits)]
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            ops.append(("cz", [i, j]))
    return ops


def _staircase_cz(n_qubits):
    ops = [("h", [q]) for q in range(n_qubits)]
    for gap in range(1, n_qubits):
        for i in range(n_qubits - gap):
            ops.append(("cz", [i, i + gap]))
    return ops


class TestRandomLayeredCircuits:
    @pytest.mark.parametrize("n_qubits,n_layers", [
        (3, 3), (3, 6), (3, 10),
        (5, 3), (5, 6), (5, 10),
        (8, 3), (8, 6), (8, 10),
        (11, 3), (11, 6), (11, 10),
    ])
    def test_random_layered(self, n_qubits, n_layers):
        rng = np.random.RandomState(42)
        ops = _random_layered_circuit(rng, n_qubits, n_layers)
        _assert_distributions_close(ops, n_qubits)

    @pytest.mark.parametrize("seed", [0, 7, 99, 256, 1024])
    def test_random_varied_seeds_8q(self, seed):
        rng = np.random.RandomState(seed)
        ops = _random_layered_circuit(rng, 8, 6)
        _assert_distributions_close(ops, 8)


class TestGHZCircuits:
    @pytest.mark.parametrize("n_qubits,ghz_start,n_ghz,use_tree", [
        (8, 0, 4, True),
        (8, 3, 4, True),
        (11, 2, 4, True),
        (11, 2, 4, False),
        (6, 0, 6, True),
        (6, 0, 6, False),
        (10, 0, 10, True),
        (10, 0, 10, False),
        (12, 2, 8, True),
        (12, 4, 6, False),
    ])
    def test_ghz(self, n_qubits, ghz_start, n_ghz, use_tree):
        ops = _ghz_circuit(n_qubits, ghz_start, n_ghz, use_tree)
        _assert_distributions_close(ops, n_qubits)


class TestSparseQubitCircuits:
    @pytest.mark.parametrize("indices", [
        [0, 3, 7],
        [1, 5, 9, 10],
        [0, 2, 4, 6, 8],
        [0, 11],
        [3, 7, 11],
        [0, 4, 8, 12],
    ])
    def test_sparse(self, indices):
        rng = np.random.RandomState(42)
        ops, n_qubits = _sparse_qubit_circuit(rng, indices, 4)
        _assert_distributions_close(ops, n_qubits)


class TestSingleGateFullRegister:
    @pytest.mark.parametrize("gate", ONE_QUBIT_GATES)
    def test_single_gate_6q(self, gate):
        ops = [(gate, [q]) for q in range(6)]
        _assert_distributions_close(ops, 6)

    @pytest.mark.parametrize("gate", ONE_QUBIT_GATES)
    def test_single_gate_10q(self, gate):
        ops = [(gate, [q]) for q in range(10)]
        _assert_distributions_close(ops, 10)


class TestThreeQubitGates:
    @pytest.mark.parametrize("n_qubits", [3, 4, 5, 7, 9])
    def test_three_qubit_gates(self, n_qubits):
        rng = np.random.RandomState(42)
        ops = _random_layered_circuit(
            rng, n_qubits, 3,
            gate_set=ONE_QUBIT_GATES + TWO_QUBIT_GATES + THREE_QUBIT_GATES,
        )
        _assert_distributions_close(ops, n_qubits)


class TestBellPairs:
    @pytest.mark.parametrize("q0,q1,n_qubits", [
        (0, 1, 2),
        (0, 1, 6),
        (0, 5, 6),
        (2, 4, 8),
        (0, 7, 8),
        (3, 9, 10),
        (0, 11, 12),
    ])
    def test_bell(self, q0, q1, n_qubits):
        ops = _bell_pair(q0, q1, n_qubits)
        _assert_distributions_close(ops, n_qubits)


class TestQFTCircuits:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_qft_full_register(self, n_qubits):
        ops = _qft_circuit(n_qubits)
        _assert_distributions_close(ops, n_qubits)

    @pytest.mark.parametrize("targets,n_qubits", [
        ([1, 2, 3], 6),
        ([0, 2, 4], 6),
        ([2, 3, 4, 5], 8),
    ])
    def test_qft_partial_register(self, targets, n_qubits):
        ops = _qft_circuit(n_qubits, targets)
        _assert_distributions_close(ops, n_qubits)

    def test_qft_after_x_gates(self):
        ops = [("x", [0]), ("x", [2])]
        ops += _qft_circuit(4)
        _assert_distributions_close(ops, 4)


class TestGroverCircuits:
    @pytest.mark.parametrize("n_qubits,marked", [
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 5), (3, 7),
    ])
    def test_grover(self, n_qubits, marked):
        ops = _grover_oracle_diffuser(n_qubits, marked)
        _assert_distributions_close(ops, n_qubits)


class TestSwapNetwork:
    @pytest.mark.parametrize("n_qubits,rounds", [
        (4, 1), (4, 2), (4, 3),
        (6, 1), (6, 2),
        (8, 1), (8, 2),
        (5, 1), (5, 2),
    ])
    def test_swap_network(self, n_qubits, rounds):
        ops = _swap_network(n_qubits, rounds)
        _assert_distributions_close(ops, n_qubits)

    def test_swap_network_with_initial_state(self):
        ops = [("x", [0]), ("h", [1])]
        ops += _swap_network(4, 2)
        _assert_distributions_close(ops, 4)


class TestRepeatedGate:
    @pytest.mark.parametrize("gate,n_qubits,reps", [
        ("h", 4, 1), ("h", 4, 2), ("h", 4, 3),
        ("x", 6, 1), ("x", 6, 2),
        ("s", 4, 1), ("s", 4, 4),
        ("t", 5, 1), ("t", 5, 8),
        ("z", 3, 1), ("z", 3, 2),
    ])
    def test_repeated(self, gate, n_qubits, reps):
        ops = _repeated_gate_circuit(gate, n_qubits, reps)
        _assert_distributions_close(ops, n_qubits)


class TestReverseCNOTLadder:
    @pytest.mark.parametrize("n_qubits", [3, 4, 5, 6, 8, 10])
    def test_reverse_cnot(self, n_qubits):
        ops = _reverse_cnot_ladder(n_qubits)
        _assert_distributions_close(ops, n_qubits)


class TestWStateApprox:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
    def test_w_state(self, n_qubits):
        ops = _w_state_approx(n_qubits)
        _assert_distributions_close(ops, n_qubits)


class TestCheckerboardCircuit:
    @pytest.mark.parametrize("n_qubits,n_layers", [
        (4, 2), (4, 4), (4, 6),
        (6, 2), (6, 4),
        (8, 2), (8, 3),
    ])
    def test_checkerboard(self, n_qubits, n_layers):
        ops = _checkerboard_circuit(n_qubits, n_layers)
        _assert_distributions_close(ops, n_qubits)


class TestLongRangeCNOT:
    @pytest.mark.parametrize("n_qubits,stride", [
        (4, 2), (4, 3),
        (6, 2), (6, 3), (6, 4),
        (8, 2), (8, 3), (8, 4),
        (10, 3), (10, 5),
        (12, 4), (12, 6),
    ])
    def test_long_range(self, n_qubits, stride):
        ops = _long_range_cnot_circuit(n_qubits, stride)
        _assert_distributions_close(ops, n_qubits)


class TestCascadeCircuit:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6, 8])
    def test_cascade(self, n_qubits):
        ops = _cascade_circuit(n_qubits)
        _assert_distributions_close(ops, n_qubits)


class TestIdentitySandwich:
    def test_sandwich_bell(self):
        inner = [("h", [0]), ("cnot", [0, 1])]
        ops = _identity_sandwich(4, inner)
        _assert_distributions_close(ops, 4)

    def test_sandwich_ghz(self):
        inner = _ghz_circuit(6, 0, 6, True)
        ops = _identity_sandwich(6, inner)
        _assert_distributions_close(ops, 6)

    def test_sandwich_random(self):
        rng = np.random.RandomState(42)
        inner = _random_layered_circuit(rng, 5, 3)
        ops = _identity_sandwich(5, inner)
        _assert_distributions_close(ops, 5)


class TestAlternatingBasis:
    @pytest.mark.parametrize("n_qubits,n_layers", [
        (4, 1), (4, 2), (4, 3),
        (6, 1), (6, 2),
        (8, 1), (8, 2),
    ])
    def test_alternating(self, n_qubits, n_layers):
        ops = _alternating_basis_circuit(n_qubits, n_layers)
        _assert_distributions_close(ops, n_qubits)


class TestDenseEntangle:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
    def test_dense(self, n_qubits):
        ops = _dense_entangle_circuit(n_qubits)
        _assert_distributions_close(ops, n_qubits)


class TestStaircaseCZ:
    @pytest.mark.parametrize("n_qubits", [3, 4, 5, 6, 8])
    def test_staircase(self, n_qubits):
        ops = _staircase_cz(n_qubits)
        _assert_distributions_close(ops, n_qubits)


class TestComposedCircuits:
    def test_qft_then_inverse(self):
        fwd = _qft_circuit(4)
        rev = list(reversed([(g, t) for g, t in fwd]))
        _assert_distributions_close(fwd + rev, 4)

    def test_ghz_then_disentangle(self):
        ops = _ghz_circuit(6, 0, 6, False)
        for i in range(5, 0, -1):
            ops.append(("cnot", [i - 1, i]))
        ops.append(("h", [0]))
        _assert_distributions_close(ops, 6)

    def test_bell_into_swap_network(self):
        ops = _bell_pair(0, 1, 6)
        ops += _swap_network(6, 2)
        _assert_distributions_close(ops, 6)

    def test_cascade_then_reverse_cnot(self):
        ops = _cascade_circuit(5)
        ops += _reverse_cnot_ladder(5)
        _assert_distributions_close(ops, 5)

    def test_checkerboard_then_dense(self):
        ops = _checkerboard_circuit(4, 2)
        ops += _dense_entangle_circuit(4)
        _assert_distributions_close(ops, 4)

    def test_long_range_then_staircase(self):
        ops = _long_range_cnot_circuit(6, 3)
        ops += _staircase_cz(6)
        _assert_distributions_close(ops, 6)


class TestEmptyCircuitBaseline:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 6, 8, 10, 12])
    def test_empty_circuit_all_zero(self, n_qubits):
        expected = np.zeros(2**n_qubits)
        expected[0] = 1.0
        _assert_exact_probabilities([], n_qubits, expected)

    @pytest.mark.parametrize("n_qubits", [1, 2, 4, 6, 8])
    def test_identity_only_all_zero(self, n_qubits):
        ops = [("i", [q]) for q in range(n_qubits)]
        expected = np.zeros(2**n_qubits)
        expected[0] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)


class TestDeterministicOutputs:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 6, 8])
    def test_x_on_all_qubits(self, n_qubits):
        ops = [("x", [q]) for q in range(n_qubits)]
        expected = np.zeros(2**n_qubits)
        expected[-1] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("q,n_qubits", [
        (0, 4), (1, 4), (2, 4), (3, 4),
        (0, 8), (4, 8), (7, 8),
    ])
    def test_x_on_single_qubit(self, q, n_qubits):
        ops = [("x", [q])]
        expected = np.zeros(2**n_qubits)
        expected[1 << (n_qubits - 1 - q)] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 6, 8])
    def test_z_on_zero_state_unchanged(self, n_qubits):
        ops = [("z", [q]) for q in range(n_qubits)]
        expected = np.zeros(2**n_qubits)
        expected[0] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("ctrl,tgt,n_qubits", [
        (0, 1, 2), (0, 1, 4), (2, 3, 4), (0, 3, 4),
        (0, 1, 8), (3, 7, 8),
    ])
    def test_cnot_on_zero_state_unchanged(self, ctrl, tgt, n_qubits):
        ops = [("cnot", [ctrl, tgt])]
        expected = np.zeros(2**n_qubits)
        expected[0] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("ctrl,tgt,n_qubits", [
        (0, 1, 2), (0, 1, 4), (2, 3, 4), (0, 3, 4),
        (0, 1, 8), (3, 7, 8),
    ])
    def test_cnot_flips_target_when_control_set(self, ctrl, tgt, n_qubits):
        ops = [("x", [ctrl]), ("cnot", [ctrl, tgt])]
        expected = np.zeros(2**n_qubits)
        ctrl_bit = 1 << (n_qubits - 1 - ctrl)
        tgt_bit = 1 << (n_qubits - 1 - tgt)
        expected[ctrl_bit | tgt_bit] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)


class TestGateCancellation:
    @pytest.mark.parametrize("gate,n_qubits", [
        ("h", 4), ("h", 8),
        ("x", 4), ("x", 8),
        ("z", 4), ("z", 8),
        ("y", 4),
    ])
    def test_self_inverse_cancels(self, gate, n_qubits):
        ops = [(gate, [q]) for q in range(n_qubits)]
        ops += [(gate, [q]) for q in range(n_qubits)]
        expected = np.zeros(2**n_qubits)
        expected[0] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("n_qubits", [2, 4, 6])
    def test_s_fourth_power_is_identity(self, n_qubits):
        ops = [("h", [q]) for q in range(n_qubits)]
        for _ in range(4):
            ops += [("s", [q]) for q in range(n_qubits)]
        ref = [("h", [q]) for q in range(n_qubits)]
        np.testing.assert_allclose(
            _reference_probabilities(ops, n_qubits),
            _reference_probabilities(ref, n_qubits),
            atol=1e-12,
        )

    @pytest.mark.parametrize("n_qubits", [2, 4, 6])
    def test_t_eighth_power_is_identity(self, n_qubits):
        ops = [("h", [q]) for q in range(n_qubits)]
        for _ in range(8):
            ops += [("t", [q]) for q in range(n_qubits)]
        ref = [("h", [q]) for q in range(n_qubits)]
        np.testing.assert_allclose(
            _reference_probabilities(ops, n_qubits),
            _reference_probabilities(ref, n_qubits),
            atol=1e-12,
        )

    @pytest.mark.parametrize("ctrl,tgt,n_qubits", [
        (0, 1, 2), (0, 1, 6), (2, 5, 6), (0, 5, 6),
    ])
    def test_cnot_self_inverse(self, ctrl, tgt, n_qubits):
        ops = [("h", [q]) for q in range(n_qubits)]
        ops += [("cnot", [ctrl, tgt]), ("cnot", [ctrl, tgt])]
        ref = [("h", [q]) for q in range(n_qubits)]
        np.testing.assert_allclose(
            _reference_probabilities(ops, n_qubits),
            _reference_probabilities(ref, n_qubits),
            atol=1e-12,
        )

    @pytest.mark.parametrize("q0,q1,n_qubits", [
        (0, 1, 2), (0, 1, 6), (2, 5, 6),
    ])
    def test_swap_self_inverse(self, q0, q1, n_qubits):
        ops = [("h", [q]) for q in range(n_qubits)]
        ops += [("swap", [q0, q1]), ("swap", [q0, q1])]
        ref = [("h", [q]) for q in range(n_qubits)]
        np.testing.assert_allclose(
            _reference_probabilities(ops, n_qubits),
            _reference_probabilities(ref, n_qubits),
            atol=1e-12,
        )


class TestIsolated2QubitGates:
    @pytest.mark.parametrize("gate", TWO_QUBIT_GATES)
    @pytest.mark.parametrize("q0,q1,n_qubits", [
        (0, 1, 2),
        (0, 1, 6), (0, 5, 6), (2, 4, 6), (3, 5, 6),
        (0, 1, 8), (0, 7, 8), (3, 6, 8), (1, 6, 8),
    ])
    def test_isolated_gate(self, gate, q0, q1, n_qubits):
        ops = [("h", [q]) for q in range(n_qubits)]
        ops.append((gate, [q0, q1]))
        _assert_distributions_close(ops, n_qubits)



class TestCNOTAllPairsStress:
    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_cnot_all_pairs_exact(self, n_qubits):
        for ctrl in range(n_qubits):
            for tgt in range(n_qubits):
                if ctrl == tgt:
                    continue
                ops = [("x", [ctrl]), ("cnot", [ctrl, tgt])]
                expected = np.zeros(2**n_qubits)
                ctrl_bit = 1 << (n_qubits - 1 - ctrl)
                tgt_bit = 1 << (n_qubits - 1 - tgt)
                expected[ctrl_bit | tgt_bit] = 1.0
                actual = _reference_probabilities(ops, n_qubits)
                np.testing.assert_allclose(
                    actual, expected, atol=1e-12,
                    err_msg=f"cnot({ctrl},{tgt}) on {n_qubits}q",
                )

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_cnot_all_pairs_braket_exact(self, n_qubits):
        for ctrl in range(n_qubits):
            for tgt in range(n_qubits):
                if ctrl == tgt:
                    continue
                ops = [("x", [ctrl]), ("cnot", [ctrl, tgt])]
                _assert_reference_matches_braket_exact(ops, n_qubits)

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_cnot_all_pairs_with_superposition(self, n_qubits):
        for ctrl in range(n_qubits):
            for tgt in range(n_qubits):
                if ctrl == tgt:
                    continue
                ops = [("h", [ctrl]), ("cnot", [ctrl, tgt])]
                _assert_reference_matches_braket_exact(ops, n_qubits)


class TestReversedQubitOrdering:
    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_cnot_high_to_low(self, n_qubits):
        for gap in range(1, n_qubits):
            ctrl = n_qubits - 1
            tgt = ctrl - gap
            if tgt < 0:
                continue
            ops = [("x", [ctrl]), ("cnot", [ctrl, tgt])]
            expected = np.zeros(2**n_qubits)
            ctrl_bit = 1 << (n_qubits - 1 - ctrl)
            tgt_bit = 1 << (n_qubits - 1 - tgt)
            expected[ctrl_bit | tgt_bit] = 1.0
            _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_cnot_high_to_low_braket(self, n_qubits):
        for gap in range(1, n_qubits):
            ctrl = n_qubits - 1
            tgt = ctrl - gap
            if tgt < 0:
                continue
            ops = [("h", [ctrl]), ("cnot", [ctrl, tgt])]
            _assert_reference_matches_braket_exact(ops, n_qubits)

    @pytest.mark.parametrize("ctrl,tgt,n_qubits", [
        (1, 0, 2), (3, 0, 4), (5, 0, 6), (7, 0, 8),
        (5, 2, 6), (7, 3, 8), (6, 1, 8), (4, 2, 6),
        (3, 1, 6), (7, 4, 8), (11, 2, 12), (9, 3, 12),
    ])
    def test_reversed_pair_exact(self, ctrl, tgt, n_qubits):
        ops = [("h", [ctrl]), ("cnot", [ctrl, tgt])]
        _assert_reference_matches_braket_exact(ops, n_qubits)

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_cz_all_orderings(self, n_qubits):
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i == j:
                    continue
                ops = [("h", [q]) for q in range(n_qubits)]
                ops.append(("cz", [i, j]))
                _assert_reference_matches_braket_exact(ops, n_qubits)


class TestCSwapCircuits:
    @pytest.mark.parametrize("ctrl,t0,t1,n_qubits", [
        (0, 1, 2, 3),
        (0, 1, 2, 6),
        (0, 2, 4, 6),
        (1, 3, 5, 6),
        (0, 1, 2, 8),
        (0, 3, 7, 8),
        (2, 4, 6, 8),
        (7, 3, 1, 8),
    ])
    def test_cswap_control_off(self, ctrl, t0, t1, n_qubits):
        ops = [("x", [t0]), ("cswap", [ctrl, t0, t1])]
        expected = np.zeros(2**n_qubits)
        expected[1 << (n_qubits - 1 - t0)] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("ctrl,t0,t1,n_qubits", [
        (0, 1, 2, 3),
        (0, 1, 2, 6),
        (0, 2, 4, 6),
        (1, 3, 5, 6),
        (0, 1, 2, 8),
        (0, 3, 7, 8),
        (2, 4, 6, 8),
        (7, 3, 1, 8),
    ])
    def test_cswap_control_on(self, ctrl, t0, t1, n_qubits):
        ops = [("x", [ctrl]), ("x", [t0]), ("cswap", [ctrl, t0, t1])]
        expected = np.zeros(2**n_qubits)
        ctrl_bit = 1 << (n_qubits - 1 - ctrl)
        t1_bit = 1 << (n_qubits - 1 - t1)
        expected[ctrl_bit | t1_bit] = 1.0
        _assert_exact_probabilities(ops, n_qubits, expected)

    @pytest.mark.parametrize("ctrl,t0,t1,n_qubits", [
        (0, 1, 2, 3), (0, 1, 2, 6), (0, 3, 7, 8), (2, 4, 6, 8),
    ])
    def test_cswap_braket_exact(self, ctrl, t0, t1, n_qubits):
        ops = [("x", [ctrl]), ("x", [t0]), ("cswap", [ctrl, t0, t1])]
        _assert_reference_matches_braket_exact(ops, n_qubits)

    @pytest.mark.parametrize("ctrl,t0,t1,n_qubits", [
        (0, 1, 2, 3), (0, 1, 2, 6), (0, 3, 7, 8),
    ])
    def test_cswap_self_inverse(self, ctrl, t0, t1, n_qubits):
        ops = [("h", [q]) for q in range(n_qubits)]
        ops += [("cswap", [ctrl, t0, t1]), ("cswap", [ctrl, t0, t1])]
        ref = [("h", [q]) for q in range(n_qubits)]
        np.testing.assert_allclose(
            _reference_probabilities(ops, n_qubits),
            _reference_probabilities(ref, n_qubits),
            atol=1e-12,
        )
