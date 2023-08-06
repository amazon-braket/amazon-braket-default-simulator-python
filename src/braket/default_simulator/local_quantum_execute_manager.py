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
from typing import Union

from braket.task_result import (
    AnalogHamiltonianSimulationTaskResult,
    AnnealingTaskResult,
    GateModelTaskResult,
)

from braket.simulator.quantum_task import QuantumExecuteManager


class LocalQuantumExecuteManager(QuantumExecuteManager):
    """A task containing the results of a local simulation.

    Since this class is instantiated with the results, cancel() and run_async() are unsupported.
    """

    def __init__(self, simulator, *args, **kwargs):
        self.simulator = simulator
        self.args = list(args)
        self.kwargs = kwargs

    def run(
        self,
    ) -> Union[GateModelTaskResult, AnnealingTaskResult, AnalogHamiltonianSimulationTaskResult,]:
        return self.simulator.run(*self.args, **self.kwargs)

    def cancel(self) -> None:
        """Cancel the quantum task."""
        raise NotImplementedError("LocalQuantumTask does not support cancelling")
