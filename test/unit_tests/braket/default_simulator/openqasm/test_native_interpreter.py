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

import pytest

from braket.default_simulator import StateVectorSimulation
from braket.default_simulator.openqasm.native_interpreter import NativeInterpreter


@pytest.mark.parametrize(
    "reset_instructions",
    (
        "for int q in [0:2 - 1] {\n    reset __qubits__[q];\n}",
        "array[int[32], 2] __arr_0__ = {0, 1};\nfor int q in __arr_0__ {\n    reset __qubits__[q];\n}",
        "reset __qubits__[0];\nreset __qubits__[1];",
        "reset __qubits__;",
    ),
)
def test_reset(reset_instructions):
    qasm = f"""
    OPENQASM 3.0;
    qubit[2] __qubits__;
    x __qubits__[0];
    x __qubits__[1];
    {reset_instructions}
    bit[2] __bit_0__ = "00";
    __bit_0__[0] = measure __qubits__[0];
    __bit_0__[1] = measure __qubits__[1];
    """

    result = NativeInterpreter(StateVectorSimulation(0, 0, 1)).simulate(qasm)
    assert result["__bit_0__"] == ["00"]
