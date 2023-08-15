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

from abc import ABC, abstractmethod
from typing import Union

from braket.task_result import (
    AnalogHamiltonianSimulationTaskResult,
    AnnealingTaskResult,
    GateModelTaskResult,
)


class ExecutionManager(ABC):
    """Manages the execution of a quantum program."""

    def cancel(self) -> None:
        """Cancel the quantum task."""
        raise NotImplementedError

    def state(self) -> str:
        """Get the state of the quantum task.
        Returns:
            str: State of the quantum task.
        """
        raise NotImplementedError

    @abstractmethod
    def result(
        self,
        *args,
        **kwargs,
    ) -> Union[GateModelTaskResult, AnnealingTaskResult, AnalogHamiltonianSimulationTaskResult,]:
        """Get the quantum task result.
        Returns:
            Union[GateModelQuantumTaskResult, AnnealingQuantumTaskResult, PhotonicModelQuantumTaskResult]: # noqa
            Get the quantum task result.
        """
