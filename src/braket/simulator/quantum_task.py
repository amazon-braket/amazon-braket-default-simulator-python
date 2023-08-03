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
from typing import Any, Dict, Union

from braket.task_result import (
    AnalogHamiltonianSimulationTaskResult,
    AnnealingTaskResult,
    GateModelTaskResult,
)


class QuantumExecuteManager(ABC):
    """An abstraction over a quantum task on a quantum device."""

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
    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[GateModelTaskResult, AnnealingTaskResult, AnalogHamiltonianSimulationTaskResult,]:
        """Get the quantum task result.
        Returns:
            Union[GateModelQuantumTaskResult, AnnealingQuantumTaskResult, PhotonicModelQuantumTaskResult]: # noqa
            Get the quantum task result.
        """

    def metadata(self, use_cached_value: bool = False) -> Dict[str, Any]:
        """
        Get task metadata.

        Args:
            use_cached_value (bool): If True, uses the value retrieved from the previous
                request. Default is False.

        Returns:
            Dict[str, Any]: The metadata regarding the task. If `use_cached_value` is True,
            then the value retrieved from the most recent request is used.
        """
