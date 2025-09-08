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

from braket.default_simulator.linalg_utils import (
    _apply_rx_gate_small,
    _apply_rx_gate_large,
    _apply_ry_gate_small,
    _apply_ry_gate_large,
    _apply_rz_gate_small,
    _apply_rz_gate_large,
    _apply_phase_shift_gate_small,
    _apply_phase_shift_gate_large,
    _apply_x_gate_large,
    _apply_y_gate_large,
    _apply_z_gate_large,
    _apply_s_gate_large,
    _apply_si_gate_large,
    _apply_t_gate_large,
    _apply_ti_gate_large,
    _apply_hadamard_gate_large,
    _apply_v_gate_large,
    _apply_vi_gate_large,
    _apply_single_qubit_gate,
    multiply_matrix,
    QuantumGateDispatcher,
)


def rx_matrix(angle):
    """Generate RX rotation matrix."""
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_matrix(angle):
    """Generate RY rotation matrix."""
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(angle):
    """Generate RZ rotation matrix."""
    return np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=complex)


def phase_shift_matrix(angle):
    """Generate phase shift matrix."""
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)


def x_matrix():
    """Generate X (Pauli-X) matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def y_matrix():
    """Generate Y (Pauli-Y) matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def z_matrix():
    """Generate Z (Pauli-Z) matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def s_matrix():
    """Generate S matrix."""
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def si_matrix():
    """Generate S† matrix."""
    return np.array([[1, 0], [0, -1j]], dtype=complex)


def t_matrix():
    """Generate T matrix."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def ti_matrix():
    """Generate T† matrix."""
    return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def hadamard_matrix():
    """Generate Hadamard matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def v_matrix():
    """Generate V (√X) matrix."""
    return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)


def vi_matrix():
    """Generate V† (√X†) matrix."""
    return np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]], dtype=complex)


rotation_gate_testdata = [
    (
        _apply_rx_gate_small,
        _apply_rx_gate_large,
        rx_matrix,
        np.pi / 2,
        0,
        [0.70710678, -0.70710678j],
    ),
    (_apply_rx_gate_small, _apply_rx_gate_large, rx_matrix, np.pi, 0, [0, -1j]),
    (
        _apply_rx_gate_small,
        _apply_rx_gate_large,
        rx_matrix,
        np.pi / 4,
        0,
        [0.92387953, -0.38268343j],
    ),
    (_apply_ry_gate_small, _apply_ry_gate_large, ry_matrix, np.pi / 2, 0, [0.70710678, 0.70710678]),
    (_apply_ry_gate_small, _apply_ry_gate_large, ry_matrix, np.pi, 0, [0, 1]),
    (_apply_ry_gate_small, _apply_ry_gate_large, ry_matrix, np.pi / 4, 0, [0.92387953, 0.38268343]),
    (
        _apply_rz_gate_small,
        _apply_rz_gate_large,
        rz_matrix,
        np.pi / 2,
        0,
        [0.70710678 - 0.70710678j, 0],
    ),
    (_apply_rz_gate_small, _apply_rz_gate_large, rz_matrix, np.pi, 0, [-1j, 0]),
    (
        _apply_rz_gate_small,
        _apply_rz_gate_large,
        rz_matrix,
        np.pi / 4,
        0,
        [0.92387953 - 0.38268343j, 0],
    ),
    (
        _apply_phase_shift_gate_small,
        _apply_phase_shift_gate_large,
        phase_shift_matrix,
        np.pi / 2,
        0,
        [1, 0],
    ),
    (
        _apply_phase_shift_gate_small,
        _apply_phase_shift_gate_large,
        phase_shift_matrix,
        np.pi,
        0,
        [1, 0],
    ),
    (
        _apply_phase_shift_gate_small,
        _apply_phase_shift_gate_large,
        phase_shift_matrix,
        np.pi / 4,
        0,
        [1, 0],
    ),
]

multi_qubit_rotation_testdata = [
    (rx_matrix, np.pi / 2, 1, 2, [1, 0, 0, 0], [0.70710678, -0.70710678j, 0, 0]),
    (rx_matrix, np.pi / 2, 1, 2, [0, 1, 0, 0], [-0.70710678j, 0.70710678, 0, 0]),
    (ry_matrix, np.pi / 2, 0, 2, [1, 0, 0, 0], [0.70710678, 0, 0.70710678, 0]),
    (rz_matrix, np.pi / 2, 0, 2, [0, 0, 1, 0], [0, 0, 0.70710678 + 0.70710678j, 0]),
    (phase_shift_matrix, np.pi / 2, 1, 2, [0, 1, 0, 0], [0, 1j, 0, 0]),
]

gate_dispatch_testdata = [
    (rx_matrix, np.pi / 2, "RX gate"),
    (rx_matrix, np.pi / 4, "RX gate quarter"),
    (ry_matrix, np.pi / 2, "RY gate"),
    (ry_matrix, np.pi / 3, "RY gate third"),
    (rz_matrix, np.pi / 2, "RZ gate"),
    (rz_matrix, 2 * np.pi / 3, "RZ gate 2pi/3"),
    (phase_shift_matrix, np.pi / 2, "Phase shift"),
    (phase_shift_matrix, 0.15, "Phase shift small"),
]


@pytest.mark.parametrize(
    "func_small, func_large, matrix_func, angle, target, expected", rotation_gate_testdata
)
def test_rotation_gates_single_qubit(func_small, func_large, matrix_func, angle, target, expected):
    """Test rotation gate functions for single qubit operations."""
    matrix = matrix_func(angle)

    state = np.array([1, 0], dtype=complex).reshape(2)
    out = np.zeros_like(state, dtype=complex)
    result, _ = func_small(state, matrix, target, out)
    assert np.allclose(result.flatten(), expected, atol=1e-7)

    state = np.array([1, 0], dtype=complex).reshape(2)
    out = np.zeros_like(state, dtype=complex)
    result, _ = func_large(state, matrix, target, out)
    assert np.allclose(result.flatten(), expected, atol=1e-7)


single_qubit_large_gate_testdata = [
    (x_matrix(), [0, 1]),
    (y_matrix(), [0, 1j]),
    (z_matrix(), [1, 0]),
    (s_matrix(), [1, 0]),
    (si_matrix(), [1, 0]),
    (t_matrix(), [1, 0]),
    (ti_matrix(), [1, 0]),
    (hadamard_matrix(), [0.70710678, 0.70710678]),
    (v_matrix(), [0.5 + 0.5j, 0.5 - 0.5j]),
    (vi_matrix(), [0.5 - 0.5j, 0.5 + 0.5j]),
    (rx_matrix(np.pi / 2), [0.70710678, -0.70710678j]),
    (rx_matrix(np.pi), [0, -1j]),
    (ry_matrix(np.pi / 2), [0.70710678, 0.70710678]),
    (ry_matrix(np.pi), [0, 1]),
    (rz_matrix(np.pi / 2), [0.70710678 - 0.70710678j, 0]),
    (rz_matrix(np.pi), [-1j, 0]),
    (phase_shift_matrix(np.pi / 2), [1, 0]),
    (phase_shift_matrix(np.pi), [1, 0]),
]


@pytest.mark.parametrize("matrix, expected", single_qubit_large_gate_testdata)
def test_single_qubit_large_gates_through_dispatcher(matrix, expected):
    """Test single qubit large gate implementations through dispatcher."""
    qubit_count = 12
    state_size = 2**qubit_count

    computational_basis_state = np.zeros(state_size, dtype=complex)
    computational_basis_state[0] = 1.0
    state = computational_basis_state.reshape([2] * qubit_count)
    out = np.zeros_like(state, dtype=complex)
    result, _ = _apply_single_qubit_gate(state, matrix, 0, out)

    result_flat = result.flatten()
    assert np.allclose(result_flat[[0, 2048]], expected, atol=1e-7)

    superposition_state = np.zeros(state_size, dtype=complex)
    superposition_state[0] = 0.6
    superposition_state[2048] = 0.8
    state = superposition_state.reshape([2] * qubit_count)
    out = np.zeros_like(state, dtype=complex)
    result, _ = _apply_single_qubit_gate(state, matrix, 0, out)

    manual_result = np.zeros(state_size, dtype=complex)
    manual_result[0] = matrix[0, 0] * 0.6 + matrix[0, 1] * 0.8
    manual_result[2048] = matrix[1, 0] * 0.6 + matrix[1, 1] * 0.8

    result_flat = result.flatten()
    assert np.allclose(result_flat[[0, 2048]], manual_result[[0, 2048]], atol=1e-7)

    original_norm = np.linalg.norm(superposition_state)
    result_norm = np.linalg.norm(result_flat)
    assert np.isclose(original_norm, result_norm, atol=1e-7)


@pytest.mark.parametrize(
    "matrix_func, angle, target, qubit_count, initial_state, expected",
    multi_qubit_rotation_testdata,
)
def test_rotation_gates_multi_qubit(
    matrix_func, angle, target, qubit_count, initial_state, expected
):
    """Test rotation gate functions for multi-qubit systems."""
    matrix = matrix_func(angle)

    state = np.array(initial_state, dtype=complex).reshape([2] * qubit_count)
    out = np.zeros_like(state, dtype=complex)

    result, _ = _apply_single_qubit_gate(state, matrix, target, out)
    assert np.allclose(result.flatten(), expected, atol=1e-7)


@pytest.mark.parametrize("matrix_func, angle, description", gate_dispatch_testdata)
def test_gate_dispatch_recognition(matrix_func, angle, description):
    """Test that _apply_single_qubit_gate correctly recognizes rotation gate patterns."""
    matrix = matrix_func(angle)

    state_small = np.array([1, 0], dtype=complex).reshape(2)
    out_small = np.zeros_like(state_small, dtype=complex)
    result_small, _ = _apply_single_qubit_gate(state_small, matrix, 0, out_small)

    state_large = np.array([1] + [0] * 1023, dtype=complex).reshape([2] * 10)
    out_large = np.zeros_like(state_large, dtype=complex)
    result_large, _ = _apply_single_qubit_gate(state_large, matrix, 0, out_large)

    large_flat = result_large.flatten()
    expected_large = [large_flat[0], large_flat[512]]

    assert np.allclose(result_small.flatten()[:2], expected_large, atol=1e-7)


def test_rx_gate_identity():
    """Test RX gate with angle 0 gives identity."""
    matrix = rx_matrix(0)
    state = np.array([0.6, 0.8], dtype=complex)
    out = np.zeros_like(state, dtype=complex)

    result, _ = _apply_rx_gate_small(state.reshape(2), matrix, 0, out.reshape(2))
    assert np.allclose(result.flatten(), state, atol=1e-7)


def test_ry_gate_identity():
    """Test RY gate with angle 0 gives identity."""
    matrix = ry_matrix(0)
    state = np.array([0.6, 0.8], dtype=complex)
    out = np.zeros_like(state, dtype=complex)

    result, _ = _apply_ry_gate_small(state.reshape(2), matrix, 0, out.reshape(2))
    assert np.allclose(result.flatten(), state, atol=1e-7)


def test_rz_gate_identity():
    """Test RZ gate with angle 0 gives identity."""
    matrix = rz_matrix(0)
    state = np.array([0.6, 0.8], dtype=complex)
    out = np.zeros_like(state, dtype=complex)

    result, _ = _apply_rz_gate_small(state.reshape(2), matrix, 0, out.reshape(2))
    assert np.allclose(result.flatten(), state, atol=1e-7)


def test_phase_shift_gate_identity():
    """Test phase shift gate with angle 0 gives identity."""
    matrix = phase_shift_matrix(0)
    state = np.array([0.6, 0.8], dtype=complex)
    out = np.zeros_like(state, dtype=complex)

    result, _ = _apply_phase_shift_gate_small(state.reshape(2), matrix, 0, out.reshape(2))
    assert np.allclose(result.flatten(), state, atol=1e-7)


def test_rotation_gates_with_multiply_matrix():
    """Test rotation gates work correctly through multiply_matrix interface."""
    angles = [np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

    for angle in angles:
        rx_mat = rx_matrix(angle)
        state = np.array([1, 0], dtype=complex).reshape(2)
        result = multiply_matrix(state, rx_mat, (0,))

        expected = rx_mat @ state.flatten()
        assert np.allclose(result.flatten(), expected, atol=1e-7)

        ry_mat = ry_matrix(angle)
        state = np.array([1, 0], dtype=complex).reshape(2)
        result = multiply_matrix(state, ry_mat, (0,))

        expected = ry_mat @ state.flatten()
        assert np.allclose(result.flatten(), expected, atol=1e-7)

        rz_mat = rz_matrix(angle)
        state = np.array([1, 0], dtype=complex).reshape(2)
        result = multiply_matrix(state, rz_mat, (0,))

        expected = rz_mat @ state.flatten()
        assert np.allclose(result.flatten(), expected, atol=1e-7)


def test_phase_shift_only_affects_one_state():
    """Test that phase shift gate only affects |1⟩ state."""
    angle = np.pi / 3
    matrix = phase_shift_matrix(angle)

    state_0 = np.array([1, 0], dtype=complex).reshape(2)
    out_0 = np.zeros_like(state_0, dtype=complex)
    result_0, _ = _apply_phase_shift_gate_small(state_0, matrix, 0, out_0)
    assert np.allclose(result_0.flatten(), [1, 0], atol=1e-7)

    state_1 = np.array([0, 1], dtype=complex).reshape(2)
    out_1 = np.zeros_like(state_1, dtype=complex)
    result_1, _ = _apply_phase_shift_gate_small(state_1, matrix, 0, out_1)
    expected = [0, np.exp(1j * angle)]
    assert np.allclose(result_1.flatten(), expected, atol=1e-7)


def test_rotation_gates_hermiticity():
    """Test that rotation gates are unitary (U† U = I)."""
    angle = np.pi / 3

    matrices = [rx_matrix(angle), ry_matrix(angle), rz_matrix(angle), phase_shift_matrix(angle)]

    for matrix in matrices:
        identity = matrix.conj().T @ matrix
        expected_identity = np.eye(2, dtype=complex)
        assert np.allclose(identity, expected_identity, atol=1e-7)


@pytest.mark.parametrize("qubit_count", [1, 2, 3, 5, 8, 12])
def test_rotation_gates_preserve_norm(qubit_count):
    """Test that rotation gates preserve state vector norm."""
    np.random.seed(42)

    state_flat = np.random.random(2**qubit_count) + 1j * np.random.random(2**qubit_count)
    state_flat = state_flat / np.linalg.norm(state_flat)
    state = state_flat.reshape([2] * qubit_count)

    angles = [np.pi / 6, np.pi / 3, np.pi / 2]
    matrices = [rx_matrix, ry_matrix, rz_matrix, phase_shift_matrix]

    for angle in angles:
        for matrix_func in matrices:
            matrix = matrix_func(angle)
            target = qubit_count // 2

            result = multiply_matrix(state, matrix, (target,))

            original_norm = np.linalg.norm(state.flatten())
            result_norm = np.linalg.norm(result.flatten())
            assert np.isclose(original_norm, result_norm, atol=1e-7)


def test_dispatcher_uses_correct_implementations():
    """Test that QuantumGateDispatcher uses appropriate implementations based on qubit count."""
    small_dispatcher = QuantumGateDispatcher(3)
    assert not small_dispatcher.use_large

    large_dispatcher = QuantumGateDispatcher(15)
    assert large_dispatcher.use_large


def test_large_implementation_edge_cases():
    """Test large implementations with edge cases and different target qubits."""
    qubit_count = 3
    state_size = 2**qubit_count

    for target_qubit in range(qubit_count):
        state_flat = np.ones(state_size, dtype=complex) / np.sqrt(state_size)
        state = state_flat.reshape([2] * qubit_count)

        rx_mat = rx_matrix(np.pi / 4)
        out = np.zeros_like(state, dtype=complex)
        result, _ = _apply_rx_gate_large(state, rx_mat, target_qubit, out)

        original_norm = np.linalg.norm(state.flatten())
        result_norm = np.linalg.norm(result.flatten())
        assert np.isclose(original_norm, result_norm, atol=1e-7)

        ry_mat = ry_matrix(np.pi / 3)
        out = np.zeros_like(state, dtype=complex)
        result, _ = _apply_ry_gate_large(state, ry_mat, target_qubit, out)

        result_norm = np.linalg.norm(result.flatten())
        assert np.isclose(original_norm, result_norm, atol=1e-7)

        rz_mat = rz_matrix(np.pi / 6)
        out = np.zeros_like(state, dtype=complex)
        result, _ = _apply_rz_gate_large(state, rz_mat, target_qubit, out)

        result_norm = np.linalg.norm(result.flatten())
        assert np.isclose(original_norm, result_norm, atol=1e-7)

        phase_mat = phase_shift_matrix(np.pi / 5)
        out = np.zeros_like(state, dtype=complex)
        result, _ = _apply_phase_shift_gate_large(state, phase_mat, target_qubit, out)

        result_norm = np.linalg.norm(result.flatten())
        assert np.isclose(original_norm, result_norm, atol=1e-7)
