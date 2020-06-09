# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Union

from braket.ir.annealing import Problem
from braket.ir.jaqcd import Program


class Paradigm(Enum):
    """ Quantum computing paradigms that are supported by the Braket SDK

    A BraketSimulator implementation must support one of these paradigms by implementing
    `supported_paradigmss` to be able to run from the Braket SDK.

    QUBIT_GATE must be supported to run programs defined by quantum gates on qubits.
    ANNEALING must be supported to run quantum annealing problems.
    """

    QUBIT_GATE = auto()
    ANNEALING = auto()


class BraketSimulator(ABC):
    """ An abstract simulator that locally runs a quantum task.

    The task can be either a circuit-based program or an annealing problem,
    specified by the given IR. The supported

    For users creating their own simulator: to register a simulator so the
    Braket SDK recognizes its name, the name and class must added as an
    entry point for "braket.simulators". This is done by adding an entry to
    entry_points in the simulator package's setup.py:

    >>> entry_points = {
    >>>     "braket.simulators": [
    >>>         "backend_name = <backend_class>"
    >>>     ]
    >>> }
    """

    @abstractmethod
    def run(self, ir: Union[Program, Problem], *args, **kwargs) -> Dict[str, Any]:
        """ Run the task specified by the given IR.

        Extra arguments will contain any additional information necessary to run the task,
        such as number of qubits.

        Args:
            ir (Union[Program, Problem]): The IR representation of the program

        Returns:
            Dict[str, Any]: A dict containing the results of the simulation.
            In order to work with braket-python-sdk, the format of the JSON dict should
            match that needed by GateModelQuantumTaskResult or AnnealingQuantumTaskResult
            from the SDK, depending on the type of task.
        """

    @property
    @abstractmethod
    def properties(self) -> Dict[str, Any]:
        """ Dict[str, Any]: Properties of the device."""

    @property
    @abstractmethod
    def supported_paradigms(self) -> FrozenSet[Paradigm]:
        """ Tuple[Paradigm, ...]: Quantum computing paradigms supported by the device."""
