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

from itertools import repeat
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .program_context import ProgramContext

from .parser.openqasm_ast import (
    ArrayLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    ClassicalType,
    FloatLiteral,
    FloatType,
    IntegerLiteral,
    IntType,
    UintType,
)

OutputValue = bool | int | float | list | None
_SCALAR_LITERALS = (BooleanLiteral, IntegerLiteral, FloatLiteral)
_NUMERIC_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


def convert_output_value(
    value: Any, var_type: ClassicalType, name: str = "output variable"
) -> OutputValue:
    """Convert a stored classical value to a schema ``OutputValue``.

    Args:
        value (Any): The stored value: an AST literal, a raw Python value,
            a list of either, or ``None``.
        var_type (ClassicalType): The evaluated declared type of the variable.
        name (str): The variable name, used in error messages. Default:
            "output variable"

    Returns:
        OutputValue: The converted value.

    Raises:
        TypeError: If the value cannot be converted for the declared type.
    """
    if value is None:
        return None
    if isinstance(var_type, BitType) and var_type.size is not None:
        return _convert_elements(value, BitType(), name)
    return _convert_scalar(value, var_type, name)


def compute_outputs_single_path(
    context: ProgramContext, measurements: list[list[str]]
) -> list[dict[str, OutputValue]]:
    """Compute per-shot output dictionaries for the single-path flow.

    Measured register elements are read from the shot's measurement row (columns
    in ``circuit.reported_measurement_slots()`` order) unless classically overwritten later.

    Args:
        context (ProgramContext): The program context after interpretation.
        measurements (list[list[str]]): The formatted measurement rows
            (``"0"``/``"1"`` per column), one row per shot.

    Returns:
        list[dict[str, OutputValue]]: One output dictionary per measurement row.
    """
    overwritten = context.overwritten_measurements
    columns = {
        (register, element): column
        for column, (register, element, _) in enumerate(
            context.circuit.reported_measurement_slots()
        )
        if (register, element) not in overwritten
    }
    rows = np.asarray(measurements, dtype=np.int8)
    per_output = []
    for name, var_type in context.output_variables.items():
        # Converted once: the program value, which also covers unmeasured elements
        value = convert_output_value(context.get_value(name), var_type, name)
        register = context.register_table.get_register(name)
        measured = (
            []
            if register is None
            else [
                (element, columns[(register, element)])
                for element in range(register.size)
                if (register, element) in columns
            ]
        )
        per_output.append(
            _measured_values(value, var_type, measured, rows)
            if measured
            else repeat(value, len(measurements))
        )
    names = list(context.output_variables)
    return [dict(zip(names, shot)) for shot in zip(*per_output)]


def compute_outputs_branched(context: ProgramContext) -> list[dict[str, OutputValue]]:
    """Compute per-shot output dictionaries for the branched (multi-path) flow.

    Each path's final variable values are its output, repeated once per shot in
    ``active_paths`` order to match the simulator's measurement rows.

    Args:
        context (ProgramContext): The branched program context after interpretation.

    Returns:
        list[dict[str, OutputValue]]: One output dictionary per shot.
    """
    output_variables = context.output_variables
    outputs = []
    for path in context.active_paths:
        values = {
            name: convert_output_value(path.get_variable(name).value, var_type, name)
            for name, var_type in output_variables.items()
        }
        outputs.extend([values] * path.shots)
    return outputs


def _measured_values(
    value: OutputValue,
    var_type: ClassicalType,
    measured: list[tuple[int, int]],
    rows: np.ndarray,
) -> list[OutputValue]:
    """Per-shot values of a bit register, with each measurement read from ``rows``.

    Args:
        value (OutputValue): The converted program value of the register. Its
            unmeasured elements are reported as is.
        var_type (ClassicalType): The declared type. An unsized ``bit`` is a
            register of size 1 reported as a scalar.
        measured (list[tuple[int, int]]): ``(register_index, column)`` for each
            measured element, in register index order. ``register_index`` is the
            index of the register element the measurement was recorded into.
        rows (np.ndarray): The measurement rows, one row per shot.

    Returns:
        list[OutputValue]: One value per shot.
    """
    register_indices, columns = zip(*measured)
    bits_per_shot = rows[:, list(columns)].tolist()
    if var_type.size is None:
        return [shot_bits[0] for shot_bits in bits_per_shot]
    if len(register_indices) == len(value):
        return bits_per_shot
    values = []
    for shot_bits in bits_per_shot:
        full = list(value)
        for register_index, bit in zip(register_indices, shot_bits, strict=True):
            full[register_index] = bit
        values.append(full)
    return values


def _convert_scalar(value: Any, var_type: ClassicalType, name: str) -> bool | int | float:
    raw = value.value if isinstance(value, _SCALAR_LITERALS) else value
    if not isinstance(raw, _NUMERIC_TYPES):
        raise TypeError(
            f"Cannot convert value {value!r} of output variable '{name}' "
            f"to type {type(var_type).__name__}."
        )
    if isinstance(var_type, BoolType):
        return bool(raw)
    if isinstance(var_type, (IntType, UintType)):
        return int(raw)
    if isinstance(var_type, FloatType):
        return float(raw)
    if isinstance(var_type, BitType):
        return int(raw)
    raise TypeError(
        f"Cannot convert value {value!r} of output variable '{name}' "
        f"to unsupported type {type(var_type).__name__}."
    )


def _convert_elements(value: Any, element_type: ClassicalType, name: str) -> list:
    if isinstance(value, ArrayLiteral):
        elements = value.values
    elif isinstance(value, (list, tuple)):
        elements = value
    else:
        raise TypeError(f"Cannot convert value {value!r} of output variable '{name}' to a list.")
    return [convert_output_value(element, element_type, name) for element in elements]
