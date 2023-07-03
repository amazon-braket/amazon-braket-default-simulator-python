import pytest
import scipy as sp

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _apply_hamiltonian,
    _get_hamiltonian,
)

operator = sp.sparse.csr_matrix(tuple([[1], [[0], [0]]]), shape=(2, 2))

operators_coefficients = (
    [operator],
    [operator],
    [operator],
    [[0.0, 1.0]],
    [[0.0, 1.0]],
    [[0.0, 1.0]],
    operator,
)

input_register = [1.0, 0.0]


@pytest.mark.parametrize(
    "index_time, operators_coefficients, warning_message",
    [
        (
            -1.0,
            operators_coefficients,
            (
                "The solver uses intermediate time value that is "
                "smaller than the minimum time value specified. "
                "The first time value of the specified range "
                "is used as an approximation."
            ),
        ),
        (
            10.0,
            operators_coefficients,
            (
                "The solver uses intermediate time value that is "
                "larger than the maximum time value specified. "
                "The final time value of the specified range "
                "is used as an approximation."
            ),
        ),
    ],
)
def test_warning_get_hamiltonian(index_time, operators_coefficients, warning_message):
    _assert_warning_is_produced_for_get_hamiltonian(
        index_time, operators_coefficients, warning_message
    )


@pytest.mark.parametrize(
    "index_time, operators_coefficients, input_register, warning_message",
    [
        (
            -1.0,
            operators_coefficients,
            input_register,
            (
                "The solver uses intermediate time value that is "
                "smaller than the minimum time value specified. "
                "The first time value of the specified range "
                "is used as an approximation."
            ),
        ),
        (
            10.0,
            operators_coefficients,
            input_register,
            (
                "The solver uses intermediate time value that is "
                "larger than the maximum time value specified. "
                "The final time value of the specified range "
                "is used as an approximation."
            ),
        ),
    ],
)
def test_warning_apply_hamiltonian(
    index_time, operators_coefficients, input_register, warning_message
):
    _assert_warning_is_produced_for_apply_hamiltonian(
        index_time, operators_coefficients, input_register, warning_message
    )


def _assert_warning_is_produced_for_get_hamiltonian(
    index_time, operators_coefficients, warning_message
):
    with pytest.warns(UserWarning) as e:
        _get_hamiltonian(index_time, operators_coefficients)
    assert warning_message in str(e[-1].message)


def _assert_warning_is_produced_for_apply_hamiltonian(
    index_time, operators_coefficients, input_register, warning_message
):
    with pytest.warns(UserWarning) as e:
        _apply_hamiltonian(index_time, operators_coefficients, input_register)
    assert warning_message in str(e[-1].message)
