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

from braket.simulator import BraketSimulator
from braket.simulator.execution_manager import ExecutionManager


class LocalExecutionManager(ExecutionManager):
    """Manages the execution of a quantum program using a local simulator."""

    def __init__(self, simulator: BraketSimulator, *args, **kwargs):
        """Initialize the LocalExecutionManager.

        Args:
            simulator (BraketSimulator): The local simulator to use for quantum program execution.
            args: Additional positional arguments for configuring the simulation.
            kwargs: Additional keyword arguments for configuring the simulation.
        """
        self.simulator = simulator
        self.args = list(args)
        self.kwargs = kwargs

    def result(
        self,
    ) -> Union[GateModelTaskResult, AnnealingTaskResult, AnalogHamiltonianSimulationTaskResult,]:
        """Get the quantum task result.

        Returns:
            Union[GateModelQuantumTaskResult, AnnealingQuantumTaskResult,
            PhotonicModelQuantumTaskResult]:
            The result of the quantum task execution using the local simulator.
        """
        return self.simulator.run(*self.args, **self.kwargs)

    def cancel(self) -> None:
        """Cancel the quantum task.

        Raises:
            NotImplementedError: LocalExecutionManager does not support cancelling.
        """
        raise NotImplementedError("LocalQuantumTask does not support cancelling")
