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

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import ClassicalRegister
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
    if isinstance(var_type, ArrayType):
        element_type = (
            var_type.base_type
            if len(var_type.dimensions) <= 1
            else ArrayType(var_type.base_type, var_type.dimensions[1:])
        )
        return _convert_elements(value, element_type, name)
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
    output_variables = context.output_variables
    overwritten = context.overwritten_measurements
    columns = {
        (register, element): column
        for column, (register, element, _) in enumerate(
            context.circuit.reported_measurement_slots()
        )
        if (register, element) not in overwritten
    }
    outputs = []
    for row in measurements:
        shot = {}
        for name, var_type in output_variables.items():
            value = context.get_value(name)
            register = context.register_table.get_register(name)
            if register is not None:
                value = _with_measured_bits(value, var_type, register, columns, row)
            shot[name] = convert_output_value(value, var_type, name)
        outputs.append(shot)
    return outputs


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


def _with_measured_bits(
    value: Any,
    var_type: ClassicalType,
    register: ClassicalRegister,
    columns: dict[tuple[ClassicalRegister, int], int],
    row: list[str],
) -> Any:
    """Return ``value`` with every measured element of ``register`` read from ``row``.

    A sized ``bit[n]`` is stored as an ``ArrayLiteral``; an unsized ``bit`` is a
    register of size 1 stored as a scalar.
    """
    bits = list(value.values) if isinstance(value, ArrayLiteral) else [value]
    for element in range(register.size):
        column = columns.get((register, element))
        if column is not None:
            bits[element] = int(row[column])
    return bits if var_type.size is not None else bits[0]


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
