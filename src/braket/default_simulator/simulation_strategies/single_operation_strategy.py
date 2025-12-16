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

from braket.default_simulator.gate_fusion import optimize_circuit
from braket.default_simulator.linalg_utils import (
    QuantumGateDispatcher,
    multiply_matrix,
)
from braket.default_simulator.operation import GateOperation

_FUSION_THRESHOLD = 10


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Applies operations to a state vector one at a time with gate fusion optimization.

    Consecutive single-qubit and two-qubit gates on the same targets are fused into
    single matrix operations before application, reducing the number of state vector
    traversals.

    Args:
        state (np.ndarray): The state vector to apply the given operations to, as a type
            (num_qubits, 0) tensor
        qubit_count (int): Unused parameter; in signature for backwards-compatibility
        operations (list[GateOperation]): The operations to apply to the state vector

    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (qubit_count, 0) tensor
    """
    if not operations:
        return state

    if len(operations) >= _FUSION_THRESHOLD:
        operations = optimize_circuit(
            operations,
            enable_single_qubit_fusion=True,
            enable_two_qubit_fusion=True,
            enable_block_fusion=False,
        )

    result = state.copy()
    temp = np.zeros_like(state, dtype=complex)

    dispatcher = QuantumGateDispatcher(state.ndim)
    for op in operations:
        targets = op.targets
        ctrl_state = op.control_state
        num_ctrl = len(ctrl_state)
        _, needs_swap = multiply_matrix(
            result,
            op.matrix,
            targets[num_ctrl:],
            targets[:num_ctrl],
            ctrl_state,
            temp,
            dispatcher,
            True,
            gate_type=getattr(op, "gate_type", None),
        )
        if needs_swap:
            result, temp = temp, result
    return result
