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

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .program_context import ProgramContext

from .parser.openqasm_ast import (
    ArrayLiteral,
    ArrayType,
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
        name (str): The variable name, used in error messages. Default: "output variable"

    Returns:
        OutputValue: The converted value.

    Raises:
        TypeError: If the value cannot be converted for the declared type.
    """
    if value is None:
        return None
    if isinstance(var_type, BitType) and var_type.size is not None:
        return _convert_elements(value, BitType(), name)
    if isinstance(var_type, ArrayType):
        element_type = (
            var_type.base_type
            if len(var_type.dimensions) <= 1
            else ArrayType(var_type.base_type, var_type.dimensions[1:])
        )
        return _convert_elements(value, element_type, name)
    return _convert_scalar(value, var_type, name)


def compute_shot_output(
    output_variables: dict[str, ClassicalType],
    base_values: dict[str, Any],
    bindings: dict[str, dict[int | None, int]],
    measurement_row: list[str] | None,
    classical_idx_columns: dict[int, int],
) -> dict[str, OutputValue]:
    """Build one shot's output dictionary.

    Args:
        output_variables (dict[str, ClassicalType]): Declared output variables,
            in declaration order.
        base_values (dict[str, Any]): Final classical state per variable
            (AST literals, raw Python values, or ``None``).
        bindings (dict[str, dict[int | None, int]]): Measurement bindings
            per variable: element index (``None`` for a scalar bit) to
            classical measurement index.
        measurement_row (list[str] | None): This shot's measurement bits
            (``"0"``/``"1"`` per column); measurement-bound elements are
            replaced with these bits. ``None`` leaves all base values as-is.
        classical_idx_columns (dict[int, int]): Maps each measured classical
            index to its column in the measurement rows (columns are ordered
            by ascending classical index).

    Returns:
        dict[str, OutputValue]: The shot's output dictionary, with every
        declared output variable name as a key.
    """
    shot_output = {}
    for name, var_type in output_variables.items():
        value = base_values.get(name)
        variable_bindings = bindings.get(name)
        if variable_bindings and measurement_row is not None:
            value = _apply_measurement_bits(
                value, var_type, variable_bindings, measurement_row, classical_idx_columns
            )
        shot_output[name] = convert_output_value(value, var_type, name)
    return shot_output


def compute_outputs_single_path(
    context: ProgramContext, measurements: list[list[str]]
) -> list[dict[str, OutputValue]]:
    """Compute per-shot output dictionaries for the single-path flow.

    Args:
        context (ProgramContext): The program context after interpretation,
            with declared output variables and measurement bindings.
        measurements (list[list[str]]): The formatted measurement rows
            (``"0"``/``"1"`` per column), one row per shot, with columns
            ordered by ``sorted(circuit.target_classical_indices)``.

    Returns:
        list[dict[str, OutputValue]]: One output dictionary per measurement row.
    """
    output_variables = context.output_variables
    bindings = context.output_bindings
    classical_idx_columns = _column_map(context)
    base_values = {name: context.get_value(name) for name in output_variables}
    return [
        compute_shot_output(output_variables, base_values, bindings, row, classical_idx_columns)
        for row in measurements
    ]


def compute_outputs_branched(
    context: ProgramContext, measurements: list[list[str]]
) -> list[dict[str, OutputValue]]:
    """Compute per-shot output dictionaries for the branched (multi-path) flow.

    Args:
        context (ProgramContext): The branched program context after interpretation,
            with active paths and shot allocations.
        measurements (list[list[str]]): The aggregated measurement rows, one per
            shot, concatenated in path order, with each path's recorded
            mid-circuit measurement outcomes already written into the rows.

    Returns:
        list[dict[str, OutputValue]]: One output dictionary per shot.
    """
    output_variables = context.output_variables
    bindings = context.output_bindings
    classical_idx_columns = _column_map(context)
    outputs = []
    shot_offset = 0
    for path in context.active_paths:
        base_values = {name: _path_base_value(context, path, name) for name in output_variables}
        for shot_idx in range(shot_offset, shot_offset + path.shots):
            row = measurements[shot_idx] if shot_idx < len(measurements) else None
            outputs.append(
                compute_shot_output(output_variables, base_values, bindings, row, classical_idx_columns)
            )
        shot_offset += path.shots
    return outputs


def _column_map(context: ProgramContext) -> dict[int, int]:
    return {
        classical_idx: column
        for column, classical_idx in enumerate(sorted(context.circuit.target_classical_indices))
    }


def _path_base_value(context: ProgramContext, path: Any, name: str) -> Any:
    framed = path.get_variable(name)
    if framed is not None:
        return framed.value
    try:
        return context.variable_table.get_value(name)
    except KeyError:
        return None


def _convert_scalar(value: Any, var_type: ClassicalType, name: str) -> bool | int | float | None:
    raw = value.value if isinstance(value, _SCALAR_LITERALS) else value
    if raw is None:
        return None
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


def _apply_measurement_bits(
    value: Any,
    var_type: ClassicalType,
    variable_bindings: dict[int | None, int],
    measurement_row: list[str],
    classical_idx_columns: dict[int, int],
) -> Any:
    if None in variable_bindings:
        classical_idx = variable_bindings[None]
        if classical_idx in classical_idx_columns:
            return int(measurement_row[classical_idx_columns[classical_idx]])
        return value
    elements = _element_list(value, var_type)
    for element_idx, classical_idx in variable_bindings.items():
        if classical_idx in classical_idx_columns:
            elements[element_idx] = int(measurement_row[classical_idx_columns[classical_idx]])
    return elements


def _element_list(value: Any, var_type: ClassicalType) -> list:
    if isinstance(value, ArrayLiteral):
        return list(value.values)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [None] * _declared_size(var_type)


def _declared_size(var_type: ClassicalType) -> int:
    if isinstance(var_type, BitType) and var_type.size is not None:
        return var_type.size.value
    if isinstance(var_type, ArrayType):
        return var_type.dimensions[0].value
    raise TypeError(f"Type {type(var_type).__name__} has no declared size.")
