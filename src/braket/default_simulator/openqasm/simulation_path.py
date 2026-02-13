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

from __future__ import annotations

from copy import deepcopy
from typing import Any

from braket.default_simulator.operation import GateOperation


class FramedVariable:
    """Variable with frame tracking for proper scoping.

    Each variable tracks which frame (scope level) it was declared in,
    enabling correct scope restoration when exiting blocks.
    """

    def __init__(self, name: str, var_type: Any, value: Any, is_const: bool, frame_number: int):
        self._name = name
        self._var_type = var_type
        self._value = value
        self._is_const = is_const
        self._frame_number = frame_number

    @property
    def name(self) -> str:
        return self._name

    @property
    def var_type(self) -> Any:
        return self._var_type

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        self._value = new_value

    @property
    def is_const(self) -> bool:
        return self._is_const

    @property
    def frame_number(self) -> int:
        return self._frame_number


class SimulationPath:
    """A single execution path in a branched simulation.

    Each path maintains its own instruction sequence, shot allocation,
    classical variable state, measurement outcomes, and scope frame number.
    When a mid-circuit measurement causes branching, paths are deep-copied
    so that each branch evolves independently.
    """

    def __init__(
        self,
        instructions: list[GateOperation] | None = None,
        shots: int = 0,
        variables: dict[str, FramedVariable] | None = None,
        measurements: dict[int, list[int]] | None = None,
        frame_number: int = 0,
    ):
        self._instructions = instructions if instructions is not None else []
        self._shots = shots
        self._variables = variables if variables is not None else {}
        self._measurements = measurements if measurements is not None else {}
        self._frame_number = frame_number

    @property
    def instructions(self) -> list[GateOperation]:
        return self._instructions

    @property
    def shots(self) -> int:
        return self._shots

    @shots.setter
    def shots(self, value: int) -> None:
        self._shots = value

    @property
    def variables(self) -> dict[str, FramedVariable]:
        return self._variables

    @property
    def measurements(self) -> dict[int, list[int]]:
        return self._measurements

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @frame_number.setter
    def frame_number(self, value: int) -> None:
        self._frame_number = value

    def branch(self) -> SimulationPath:
        """Create a deep copy of this path for branching.

        Returns a new SimulationPath with independent copies of all mutable
        state (instructions, variables, measurements), so modifications to
        the child path do not affect the parent.
        """
        return SimulationPath(
            instructions=list(self._instructions),
            shots=self._shots,
            variables=deepcopy(self._variables),
            measurements=deepcopy(self._measurements),
            frame_number=self._frame_number,
        )

    def enter_frame(self) -> int:
        """Enter a new variable scope frame.

        Returns the previous frame number so it can be restored on exit.
        """
        previous = self._frame_number
        self._frame_number += 1
        return previous

    def exit_frame(self, previous_frame: int) -> None:
        """Exit the current variable scope frame.

        Removes all variables declared in frames newer than `previous_frame`
        and restores the frame number.
        """
        self._variables = {
            name: var for name, var in self._variables.items() if var.frame_number <= previous_frame
        }
        self._frame_number = previous_frame

    def add_instruction(self, instruction: GateOperation) -> None:
        """Append a gate operation to this path's instruction sequence."""
        self._instructions.append(instruction)

    def set_variable(self, name: str, var: FramedVariable) -> None:
        """Set a classical variable in this path's variable state."""
        self._variables[name] = var

    def get_variable(self, name: str) -> FramedVariable | None:
        """Get a classical variable from this path's variable state."""
        return self._variables.get(name)

    def record_measurement(self, qubit_idx: int, outcome: int) -> None:
        """Record a measurement outcome for a qubit on this path."""
        if qubit_idx not in self._measurements:
            self._measurements[qubit_idx] = []
        self._measurements[qubit_idx].append(outcome)
