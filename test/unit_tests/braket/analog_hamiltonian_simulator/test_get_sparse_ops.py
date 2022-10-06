import pytest
import scipy as sp
from braket.ir.ahs.program_v1 import Program

from braket.analog_hamiltonian_simulator.rydberg.constants import RYDBERG_INTERACTION_COEF
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    _get_sparse_from_dict,
    get_sparse_ops,
)
from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

a = 3
rydberg_interaction_coef = RYDBERG_INTERACTION_COEF

amplitude_1 = {"pattern": "uniform", "time_series": {"times": [0, 4e-6], "values": [10e6, 25e6]}}

detuning_1 = {
    "pattern": "uniform",
    "time_series": {"times": [0, 2e-6, 4e-6], "values": [-10e6, 25e6, 0]},
}

phase_1 = {
    "pattern": "uniform",
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [10, 20, -30, 40]},
}

shift_1 = {
    "pattern": [0.5, 1.0],
    "time_series": {"times": [0, 2e-6, 3e-6, 4e-6], "values": [1e7, 2e7, -3e7, 4e7]},
}

setup_1 = {"ahs_register": {"sites": [[0, 0], [0, 3e-6]], "filling": [1, 1]}}

program1 = convert_unit(
    Program(
        setup=setup_1,
        hamiltonian={
            "drivingFields": [{"amplitude": amplitude_1, "phase": phase_1, "detuning": detuning_1}],
            "shiftingFields": [{"magnitude": shift_1}],
        },
    )
)

configurations_1 = ["gg", "gr", "rg", "rr"]


def test_get_sparse_from_dict_1():
    dic = dict({(0, 0): 1})
    N = 3
    mat = _get_sparse_from_dict(dic, N)
    truth = sp.sparse.csr_matrix(tuple([[1], [[0], [0]]]), shape=(N, N))

    assert (mat != truth).nnz == 0


def test_get_sparse_from_dict_2():
    vals = [1, -1, 10, 11, 21, 100]
    rows = [0, 1, 3, 5, 1, 90]
    cols = [1, 4, 0, 399, 1, 23]
    dic = dict()
    for val, row, col in zip(vals, rows, cols):
        dic.update(dict({(row, col): val}))

    N = 400
    mat = _get_sparse_from_dict(dic, N)
    truth = sp.sparse.csr_matrix(tuple([vals, [rows, cols]]), shape=(N, N))

    assert (mat != truth).nnz == 0


@pytest.mark.parametrize("para", [[program1, rydberg_interaction_coef, configurations_1]])
def test_get_sparse_ops(para):
    program1, rydberg_interaction_coef, configurations_1 = para[0], para[1], para[2]
    rabi_ops, detuning_ops, interaction_op, local_detuning_ops = get_sparse_ops(
        program1, configurations_1, rydberg_interaction_coef
    )

    true_interaction_op = sp.sparse.csr_matrix(
        tuple([[rydberg_interaction_coef / (a**6)], [[3], [3]]]),
        shape=(len(configurations_1), len(configurations_1)),
    )

    true_detuning_ops = sp.sparse.csr_matrix(
        tuple([[1, 1, 2], [[1, 2, 3], [1, 2, 3]]]),
        shape=(len(configurations_1), len(configurations_1)),
    )

    true_rabi_ops = sp.sparse.csr_matrix(
        tuple([[1, 1, 1, 1], [[1, 2, 3, 3], [0, 0, 1, 2]]]),
        shape=(len(configurations_1), len(configurations_1)),
    )

    true_local_detuning_ops = sp.sparse.csr_matrix(
        tuple([[1.0, 0.5, 1.5], [[1, 2, 3], [1, 2, 3]]]),
        shape=(len(configurations_1), len(configurations_1)),
    )

    assert (true_interaction_op != interaction_op).nnz == 0
    assert len(detuning_ops) == 1
    assert (true_detuning_ops != detuning_ops[0]).nnz == 0
    assert len(rabi_ops) == 1
    assert (true_rabi_ops != rabi_ops[0]).nnz == 0
    assert len(local_detuning_ops) == 1
    assert (true_local_detuning_ops != local_detuning_ops[0]).nnz == 0
