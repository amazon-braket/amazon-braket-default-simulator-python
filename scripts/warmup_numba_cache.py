import sys
import numpy as np


def warmup_linalg_utils():
    """Pre-compile linalg_utils JIT functions."""
    print("Warming up Numba JIT cache for amazon-braket-default-simulator...")

    try:
        from braket.default_simulator.linalg_utils import (
            _apply_single_qubit_gate_large,
            _apply_diagonal_gate_large,
            _apply_cnot_large,
            _apply_swap_large,
            _apply_controlled_phase_shift_large,
            _apply_two_qubit_gate_large,
        )

        n_qubits = 12
        state_shape = tuple([2] * n_qubits)
        state = np.zeros(state_shape, dtype=complex)
        state.flat[0] = 1.0 + 0.0j
        out = np.zeros_like(state)

        matrix_1q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        _apply_single_qubit_gate_large(state, matrix_1q, 0, out)
        _apply_diagonal_gate_large(state, matrix_1q, 0, out)
        _apply_cnot_large(state, 0, 1, out)
        _apply_swap_large(state, 0, 1, out)

        phase = 1.0 + 0.0j
        controls = np.array([0], dtype=np.int64)
        _apply_controlled_phase_shift_large(state, phase, controls, 1)

        matrix_2q = np.eye(4, dtype=complex)
        _apply_two_qubit_gate_large(state, matrix_2q, 0, 1, out)

        print("✓ Numba JIT cache warmed up successfully")
        return True

    except Exception as e:
        print(f"Warning: Could not warm up Numba cache: {e}")
        print("This is not critical - functions will compile on first use")
        return False


if __name__ == "__main__":
    success = warmup_linalg_utils()
    sys.exit(0 if success else 0)
