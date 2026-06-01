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
import warnings

import numpy as np
import pytest
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.openqasm import Program as OpenQASMProgram

from braket.default_simulator import observables
from braket.default_simulator.result_types import DensityMatrix, Expectation, Probability, Variance
from braket.default_simulator.simulator import BaseLocalSimulator
from braket.default_simulator.state_vector_simulator import StateVectorSimulator


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


def _minimal_jaqcd_program() -> JaqcdProgram:
    """Build a minimal JaqcdProgram for warning-assertion tests."""
    return JaqcdProgram.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "probability", "targets": [0]}],
            }
        )
    )


def _minimal_openqasm_program() -> OpenQASMProgram:
    """Build a minimal OpenQASMProgram for warning-assertion tests."""
    return OpenQASMProgram(
        source="""
        qubit q;
        h q;
        #pragma braket result probability q
        """
    )


def test_run_jaqcd_emits_warning():
    """Submitting a JaqcdProgram via BaseLocalSimulator.run should emit a
    UserWarning telling the customer to migrate to OpenQASMProgram. The
    program still executes — this is a soft deprecation."""
    simulator = StateVectorSimulator()
    with pytest.warns(UserWarning, match="JaqcdProgram"):
        result = simulator.run(_minimal_jaqcd_program(), qubit_count=1, shots=10)
    assert result is not None


def test_run_openqasm_no_jaqcd_warning():
    """The OpenQASM path must not emit the JAQCD deprecation warning. Assert
    specifically on JAQCD-message warnings rather than erroring on all
    UserWarnings, since other unrelated warnings (e.g. the density-matrix
    noise advisory) are legitimate."""
    simulator = StateVectorSimulator()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = simulator.run(_minimal_openqasm_program(), shots=10)
    jaqcd_warnings = [x for x in w if "JaqcdProgram" in str(x.message)]
    assert jaqcd_warnings == []
    assert result is not None
