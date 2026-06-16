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

import json
import re

import numpy as np
import pytest
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.ir.openqasm.program_set_v1 import ProgramSet
from braket.ir.openqasm.program_v1 import Program

from braket.default_simulator import observables
from braket.default_simulator.density_matrix_simulator import DensityMatrixSimulator
from braket.default_simulator.result_types import DensityMatrix, Expectation, Probability, Variance
from braket.default_simulator.simulator import BaseLocalSimulator
from braket.default_simulator.state_vector_simulator import StateVectorSimulator

_TIMESTAMP_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


@pytest.mark.parametrize(
    "result_type",
    [
        Expectation(observables.PauliX([1])),
        Variance(observables.TensorProduct([observables.PauliY([0]), observables.PauliZ([1])])),
        Expectation(
            observables.TensorProduct(
                [observables.Identity([0]), observables.Hermitian(np.eye(2), [1])]
            )
        ),
        Expectation(observables.Hermitian(np.eye(4), [0, 1])),
        Variance(observables.PauliX()),
        DensityMatrix([1]),
        Probability([1]),
    ],
)
def test_validate_result_types_qubits_exist(result_type):
    BaseLocalSimulator._validate_result_types_qubits_exist([result_type], 2)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "result_type", [Expectation(observables.PauliX([1])), DensityMatrix([1]), Probability([1])]
)
def test_validate_result_types_qubits_exist_error(result_type):
    BaseLocalSimulator._validate_result_types_qubits_exist([result_type], 1)


def test_observable_hash_tensor_product():
    matrix = np.eye(4)
    obs = observables.TensorProduct(
        [observables.PauliX([0]), observables.Hermitian(matrix, [1, 2]), observables.PauliY([1])]
    )
    hash_dict = BaseLocalSimulator._observable_hash(obs)
    matrix_hash = hash_dict[1]
    assert hash_dict == {0: "PauliX", 1: matrix_hash, 2: matrix_hash, 3: "PauliY"}


def test_base_local_simulator_abstract():
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseLocalSimulator"):
        BaseLocalSimulator()


def _assert_valid_timestamps(created_at, ended_at):
    assert created_at is not None
    assert ended_at is not None
    assert _TIMESTAMP_REGEX.match(created_at)
    assert _TIMESTAMP_REGEX.match(ended_at)
    # The task cannot end before it was created.
    assert ended_at >= created_at


@pytest.mark.parametrize("simulator", [StateVectorSimulator(), DensityMatrixSimulator()])
@pytest.mark.parametrize(
    "ir, shots",
    [
        (OpenQASMProgram(source="bit[1] b;\nqubit[1] q;\nh q;\nb = measure q;"), 10),
        (OpenQASMProgram(source="qubit[1] q;\nh q;\n#pragma braket result probability"), 0),
        (
            JaqcdProgram.parse_raw(
                json.dumps(
                    {
                        "instructions": [{"type": "h", "target": 0}],
                        "results": [{"type": "probability"}],
                    }
                )
            ),
            0,
        ),
    ],
)
def test_task_metadata_timestamps(simulator, ir, shots):
    result = simulator.run(ir, shots=shots)
    _assert_valid_timestamps(result.taskMetadata.createdAt, result.taskMetadata.endedAt)


def test_task_metadata_timestamps_branched():
    """Mid-circuit measurement programs take the branched execution path."""
    mcm = """
    OPENQASM 3.0;
    bit[2] b;
    qubit[2] q;
    h q[0];
    b[0] = measure q[0];
    if (b[0]) {
        x q[1];
    }
    b[1] = measure q[1];
    """
    result = StateVectorSimulator().run(OpenQASMProgram(source=mcm), shots=10)
    _assert_valid_timestamps(result.taskMetadata.createdAt, result.taskMetadata.endedAt)


def test_program_set_task_metadata_timestamps():
    program = Program(source="bit[2] b;\nqubit[2] q;\nx q;\nb = measure q;")
    program_set = ProgramSet(programs=[program, program])
    result = StateVectorSimulator().run(program_set, shots=10)
    _assert_valid_timestamps(result.taskMetadata.createdAt, result.taskMetadata.endedAt)
