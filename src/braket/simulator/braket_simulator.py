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
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from os import cpu_count
from typing import Any, Optional, Union

from braket.device_schema import DeviceCapabilities
from braket.ir.ahs import Program as AHSProgram
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.openqasm import Program as OQ3Program
from braket.task_result import AnalogHamiltonianSimulationTaskResult, GateModelTaskResult


class BraketSimulator(ABC):
    """An abstract simulator that locally runs a quantum task.

    The task can be either a quantum circuit defined in an OpenQASM or JAQCD program,
    or an analog Hamiltonian simulation (AHS) program.

    For users creating their own simulator: to register a simulator so the
    Braket SDK recognizes its name, the name and class must be added as an
    entry point for "braket.simulators". This is done by adding an entry to
    entry_points in the simulator package's setup.py:

    >>> entry_points = {
    >>>     "braket.simulators": [
    >>>         "backend_name = <backend_class>"
    >>>     ]
    >>> }
    """

    DEVICE_ID = None

    @abstractmethod
    def run(
        self, ir: Union[OQ3Program, AHSProgram, JaqcdProgram], *args, **kwargs
    ) -> Union[GateModelTaskResult, AnalogHamiltonianSimulationTaskResult]:
        """
        Run the task specified by the given IR.

        Extra arguments will contain any additional information necessary to run the task,
        such as number of qubits.

        Args:
            ir (Union[OQ3Program, AHSProgram, JaqcdProgram]): The IR representation of the program

        Returns:
            Union[GateModelTaskResult, AnalogHamiltonianSimulationTaskResult]: An object
            representing the results of the simulation.
        """

    def run_multiple(
        self,
        payloads: Sequence[Union[OQ3Program, AHSProgram, JaqcdProgram]],
        args: Optional[Sequence[Sequence[Any]]] = None,
        kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
        max_parallel: Optional[int] = None,
    ) -> list[Union[GateModelTaskResult, AnalogHamiltonianSimulationTaskResult]]:
        """
        Run the tasks specified by the given IR payloads.

        Extra arguments will contain any additional information necessary to run the tasks,
        such as number of qubits.

        Args:
            payloads (Sequence[Union[OQ3Program, AHSProgram, JaqcdProgram]]): The IR representations
                of the programs
            args (Optional[Sequence[Sequence[Any]]]): The positional args to include with
                each payload; the nth entry of this sequence corresponds to the nth payload.
                If specified, the length of args must be equal to the length of payloads.
                Default: None.
            kwargs (Optional[Sequence[Mapping[str, Any]]]): The keyword args to include with
                each payload; the nth entry of this sequence corresponds to the nth payload.
                If specified, the length of kwargs must be equal to the length of payloads.
                Default: None.
            max_parallel (Optional[int]): The maximum number of payloads to run in parallel.
                Default is the number of CPUs.

        Returns:
            list[Union[GateModelTaskResult, AnalogHamiltonianSimulationTaskResult]]: A list of
            result objects, with the ith object being the result of the ith program.
        """
        max_parallel = max_parallel or cpu_count()
        if args and len(args) != len(payloads):
            raise ValueError("The number of arguments must equal the number of payloads.")
        if kwargs and len(kwargs) != len(payloads):
            raise ValueError("The number of keyword arguments must equal the number of payloads.")
        get_nth_args = (lambda n: args[n]) if args else lambda _: []
        get_nth_kwargs = (lambda n: kwargs[n]) if kwargs else lambda _: {}
        with Pool(min(max_parallel, len(payloads))) as pool:
            results = pool.starmap(
                self._run_wrapped,
                [(payloads[i], get_nth_args(i), get_nth_kwargs(i)) for i in range(len(payloads))],
            )
        return results

    def _run_wrapped(
        self, ir: Union[OQ3Program, AHSProgram, JaqcdProgram], args, kwargs
    ):  # pragma: no cover
        return self.run(ir, *args, **kwargs)

    @property
    @abstractmethod
    def properties(self) -> DeviceCapabilities:
        """DeviceCapabilities: Properties of the device."""
