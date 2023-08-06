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

from braket.default_simulator.local_quantum_execute_manager import LocalQuantumExecuteManager


class DummySimulator:
    def run(self, *args, **kwargs):
        return "DummyResult"


def test_local_quantum_execute_manager_run():
    # Create a DummySimulator instance for testing
    simulator = DummySimulator()

    # Instantiate the LocalQuantumExecuteManager with the DummySimulator
    quantum_manager = LocalQuantumExecuteManager(simulator, 1, 2, arg1="val1", arg2="val2")

    # Call the run method and check the result
    result = quantum_manager.run()
    assert result == "DummyResult"


def test_local_quantum_execute_manager_cancel():
    # Create a DummySimulator instance for testing
    simulator = DummySimulator()

    # Instantiate the LocalQuantumExecuteManager with the DummySimulator
    quantum_manager = LocalQuantumExecuteManager(simulator, 1, 2, arg1="val1", arg2="val2")

    # Call the cancel method and check if NotImplementedError is raised
    with pytest.raises(NotImplementedError):
        quantum_manager.cancel()
