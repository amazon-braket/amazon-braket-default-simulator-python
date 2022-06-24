import numpy as np
import pytest

from braket.default_simulator import StateVectorSimulation
from braket.default_simulator.gate_operations import (
    CV,
    CX,
    CY,
    CZ,
    ECR,
    XX,
    XY,
    YY,
    ZZ,
    CCNot,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    CSwap,
    Hadamard,
    Identity,
    ISwap,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    PSwap,
    RotX,
    RotY,
    RotZ,
    S,
    Si,
    T,
    Ti,
    V,
    Vi,
)
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.simulation_strategies.single_operation_strategy import (
    apply_operations,
)


@pytest.mark.parametrize(
    "gate_name, gate_class, num_qubits, params",
    (
        ("h", Hadamard, 1, ()),
        ("i", Identity, 1, ()),
        ("x", PauliX, 1, ()),
        ("y", PauliY, 1, ()),
        ("z", PauliZ, 1, ()),
        ("s", S, 1, ()),
        ("si", Si, 1, ()),
        ("t", T, 1, ()),
        ("ti", Ti, 1, ()),
        ("v", V, 1, ()),
        ("vi", Vi, 1, ()),
        ("rx", RotX, 1, (2,)),
        ("ry", RotY, 1, (2,)),
        ("rz", RotZ, 1, (2,)),
        ("phaseshift", PhaseShift, 1, (2,)),
        ("cnot", CX, 2, ()),
        ("iswap", ISwap, 2, ()),
        ("pswap", PSwap, 2, (2,)),
        ("xy", XY, 2, (2,)),
        ("cphaseshift", CPhaseShift, 2, (2,)),
        ("cphaseshift00", CPhaseShift00, 2, (2,)),
        ("cphaseshift01", CPhaseShift01, 2, (2,)),
        ("cphaseshift10", CPhaseShift10, 2, (2,)),
        ("cv", CV, 2, ()),
        ("cy", CY, 2, ()),
        ("cz", CZ, 2, ()),
        ("ecr", ECR, 2, ()),
        ("xx", XX, 2, (2,)),
        ("yy", YY, 2, (2,)),
        ("zz", ZZ, 2, (2,)),
        ("ccnot", CCNot, 3, ()),
        ("cswap", CSwap, 3, ()),
    ),
)
def test_gates(gate_name, gate_class, num_qubits, params):
    param_string = f"({', '.join(str(x) for x in params)})" if params else ""
    qubit_string = ", ".join(f"q[{i}]" for i in range(num_qubits))
    qasm = f"""
    qubit[{num_qubits}] q;
    {gate_name}{param_string} {qubit_string};
    """
    circuit = Interpreter().build_circuit(qasm)
    # assert correctness for each basis vector
    for i in range(2**num_qubits):
        state = np.zeros([2] * num_qubits)
        state[np.unravel_index(i, state.shape)] = 1
        oq3_state = apply_operations(state, num_qubits, circuit.instructions)
        gate_params = (range(num_qubits),) + params
        other_state = apply_operations(state, num_qubits, [gate_class(*gate_params)])
        assert np.allclose(oq3_state, other_state)
