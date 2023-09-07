import numpy as np
import pytest

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _get_lind_dict_T1,
    _get_lind_dict_T2,
    _get_ops_coefs_lind,
    _get_sparse_from_dict,
    noise_type,
)

T1_TEST = 100.0
T2_TEST = 150.0

noises_empty = {}
noises_t1 = {noise_type.T_1: T1_TEST}
noises_t2 = {noise_type.T_2: T2_TEST}
noises_t1t2 = {noise_type.T_1: T1_TEST, noise_type.T_2: T2_TEST}

configurations_1qb = ["g", "r"]
configurations_2qb = ["gg", "gr", "rg", "rr"]
configurations_3qb = ["ggg", "ggr", "grg", "grr", "rgg", "rgr", "rrg", "rrr"]

# dicts for T1 decay operators
sigma_m_dict = {(0, 1): 1.0}  # single qubit sigma_minus (|g \rangle\langle r|)
m_I_dict = {(0, 2): 1.0, (1, 3): 1.0}
I_m_dict = {(0, 1): 1.0, (2, 3): 1.0}
I_m_I_dict = {(0, 2): 1.0, (1, 3): 1.0, (4, 6): 1.0, (5, 7): 1.0}

# dicts for T2 decay operators
rr_dict = {(1, 1): 1.0}  # projector to |r\rangle
rr_I_dict = {(2, 2): 1.0, (3, 3): 1.0}
I_rr_dict = {(1, 1): 1.0, (3, 3): 1.0}
I_rr_I_dict = {(2, 2): 1.0, (3, 3): 1.0, (6, 6): 1.0, (7, 7): 1.0}
I_I_rr_dict = {(1, 1): 1.0, (3, 3): 1.0, (5, 5): 1.0, (7, 7): 1.0}


def float_dict_equal(dict1: dict, dict2: dict) -> bool:
    if len(dict1) != len(dict2):
        return False
    for key in dict1.keys():
        if key not in dict2:
            return False
        if not np.isclose(dict1[key], dict2[key]):
            return False
    return True


@pytest.mark.parametrize(
    "target, configurations, true_result",
    [
        ((0,), configurations_1qb, sigma_m_dict),
        ((0,), configurations_2qb, m_I_dict),
        ((1,), configurations_2qb, I_m_dict),
        ((1,), configurations_3qb, I_m_I_dict),
        ((0,), ["r"], {}),
    ],
)
def test_get_lind_dict_T1(target, configurations, true_result):
    result = _get_lind_dict_T1(target, configurations)
    assert float_dict_equal(result, true_result)


@pytest.mark.parametrize(
    "target, configurations, true_result",
    [
        ((0,), configurations_1qb, rr_dict),
        ((0,), configurations_2qb, rr_I_dict),
        ((1,), configurations_2qb, I_rr_dict),
        ((1,), configurations_3qb, I_rr_I_dict),
        ((2,), configurations_3qb, I_I_rr_dict),
    ],
)
def test_get_lind_dict_T2(target, configurations, true_result):
    result = _get_lind_dict_T2(target, configurations)
    assert float_dict_equal(result, true_result)


warning_message = (
    "No quantum channel noise speficied, using density matrix simulator"
    "is inefficient. Using the braket_ahs simulator is more efficient."
)


def test_get_ops_coefs_lind_validation():
    with pytest.warns(UserWarning) as e:
        _get_ops_coefs_lind([0, 1, 2, 3], configurations_2qb, {})
    assert warning_message in str(e[-1].message)


# sparse matrix for operators
m_I_sparse = _get_sparse_from_dict(m_I_dict, 4)
I_m_sparse = _get_sparse_from_dict(I_m_dict, 4)
rr_I_sparse = _get_sparse_from_dict(rr_I_dict, 4)
I_rr_sparse = _get_sparse_from_dict(I_rr_dict, 4)


@pytest.mark.parametrize(
    "targets, configurations, noises, true_ops, true_coefs",
    [
        ([0, 1], configurations_2qb, noises_t1, [m_I_sparse, I_m_sparse], [1 / T1_TEST] * 2),
        ([0, 1], configurations_2qb, noises_t2, [rr_I_sparse, I_rr_sparse], [2 / T2_TEST] * 2),
        (
            [0, 1],
            configurations_2qb,
            noises_t1t2,
            [m_I_sparse, I_m_sparse, rr_I_sparse, I_rr_sparse],
            [1 / T1_TEST] * 2 + [2 / T2_TEST] * 2,
        ),
    ],
)
def test_get_ops_coefs_lind(targets, configurations, noises, true_ops, true_coefs):
    result_ops, result_coefs = _get_ops_coefs_lind(targets, configurations, noises)

    assert len(result_ops) == len(true_ops)
    assert all([(op != true_op).nnz == 0 for op, true_op in zip(result_ops, true_ops)])
    assert all([np.isclose(coef, true_coef) for coef, true_coef in zip(result_coefs, true_coefs)])
