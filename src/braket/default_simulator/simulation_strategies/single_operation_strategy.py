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

from braket.default_simulator.linalg_utils import QuantumGateDispatcher, multiply_matrix
from braket.default_simulator.operation import GateOperation


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Applies operations to a state vector one at a time.
    Args:
        state (np.ndarray): The state vector to apply the given operations to, as a type
            (num_qubits, 0) tensor
        qubit_count (int): Unused parameter; in signature for backwards-compatibility
        operations (list[GateOperation]): The operations to apply to the state vector
    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (qubit_count, 0) tensor
    """
    result = state.copy()
    temp = np.zeros_like(state, dtype=complex)

    dispatcher = QuantumGateDispatcher(state.ndim)
    for op in operations:
        targets = op.targets
        num_ctrl = len(op.control_state)
        _, needs_swap = multiply_matrix(
            result,
            op.matrix,
            targets[num_ctrl:],
            targets[:num_ctrl],
            op.control_state,
            temp,
            dispatcher,
            True,
            gate_type=getattr(op, "gate_type"),
        )
        if needs_swap:
            result, temp = temp, result
    return result
