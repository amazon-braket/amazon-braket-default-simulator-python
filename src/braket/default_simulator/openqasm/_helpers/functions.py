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

"""
Evaluating expressions
"""
import math
from typing import Optional, Type, Union

import numpy as np
import sympy

from ..parser.openqasm_ast import (
    ArrayLiteral,
    AssignmentOperator,
    BinaryOperator,
    BitstringLiteral,
    BooleanLiteral,
    FloatLiteral,
    IntegerLiteral,
    SymbolLiteral,
    UintType,
    UnaryOperator,
)
from .casting import LiteralType, cast_to, convert_bool_array_to_string

operator_maps = {
    IntegerLiteral: {
        # returns int
        getattr(BinaryOperator, "+"): lambda x, y: IntegerLiteral(x.value + y.value),
        getattr(BinaryOperator, "-"): lambda x, y: IntegerLiteral(x.value - y.value),
        getattr(BinaryOperator, "*"): lambda x, y: IntegerLiteral(x.value * y.value),
        getattr(BinaryOperator, "/"): lambda x, y: IntegerLiteral(x.value / y.value),
        getattr(BinaryOperator, "%"): lambda x, y: IntegerLiteral(x.value % y.value),
        getattr(BinaryOperator, "**"): lambda x, y: IntegerLiteral(x.value**y.value),
        getattr(UnaryOperator, "-"): lambda x: IntegerLiteral(-x.value),
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(x.value > y.value),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(x.value < y.value),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(x.value >= y.value),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(x.value <= y.value),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(x.value == y.value),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(x.value != y.value),
    },
    FloatLiteral: {
        # returns real
        getattr(BinaryOperator, "+"): lambda x, y: FloatLiteral(x.value + y.value),
        getattr(BinaryOperator, "-"): lambda x, y: FloatLiteral(x.value - y.value),
        getattr(BinaryOperator, "*"): lambda x, y: FloatLiteral(x.value * y.value),
        getattr(BinaryOperator, "/"): lambda x, y: FloatLiteral(x.value / y.value),
        getattr(BinaryOperator, "**"): lambda x, y: FloatLiteral(x.value**y.value),
        getattr(UnaryOperator, "-"): lambda x: FloatLiteral(-x.value),
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(x.value > y.value),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(x.value < y.value),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(x.value >= y.value),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(x.value <= y.value),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(x.value == y.value),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(x.value != y.value),
    },
    SymbolLiteral: {
        # returns real
        getattr(BinaryOperator, "+"): lambda x, y: SymbolLiteral(x.value + y.value),
        getattr(BinaryOperator, "-"): lambda x, y: SymbolLiteral(x.value - y.value),
        getattr(BinaryOperator, "*"): lambda x, y: SymbolLiteral(x.value * y.value),
        getattr(BinaryOperator, "/"): lambda x, y: SymbolLiteral(x.value / y.value),
        getattr(BinaryOperator, "**"): lambda x, y: SymbolLiteral(x.value**y.value),
        getattr(UnaryOperator, "-"): lambda x: SymbolLiteral(-x.value),
    },
    BooleanLiteral: {
        # returns bool
        getattr(BinaryOperator, "&"): lambda x, y: BooleanLiteral(x.value and y.value),
        getattr(BinaryOperator, "|"): lambda x, y: BooleanLiteral(x.value or y.value),
        getattr(BinaryOperator, "^"): lambda x, y: BooleanLiteral(x.value ^ y.value),
        getattr(BinaryOperator, "&&"): lambda x, y: BooleanLiteral(x.value and y.value),
        getattr(BinaryOperator, "||"): lambda x, y: BooleanLiteral(x.value or y.value),
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(x.value > y.value),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(x.value < y.value),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(x.value >= y.value),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(x.value <= y.value),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(x.value == y.value),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(x.value != y.value),
        getattr(UnaryOperator, "!"): lambda x: BooleanLiteral(not x.value),
    },
    # Array literals are only used to store bit registers
    ArrayLiteral: {
        # returns array
        getattr(BinaryOperator, "&"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(xv.value and yv.value) for xv, yv in zip(x.values, y.values)]
        ),
        getattr(BinaryOperator, "|"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(xv.value or yv.value) for xv, yv in zip(x.values, y.values)]
        ),
        getattr(BinaryOperator, "^"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(xv.value ^ yv.value) for xv, yv in zip(x.values, y.values)]
        ),
        getattr(BinaryOperator, "<<"): lambda x, y: ArrayLiteral(
            x.values[y.value :] + [BooleanLiteral(False) for _ in range(y.value)]
        ),
        getattr(BinaryOperator, ">>"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(False) for _ in range(y.value)] + x.values[: len(x.values) - y.value]
        ),
        getattr(UnaryOperator, "~"): lambda x: ArrayLiteral(
            [BooleanLiteral(not v.value) for v in x.values]
        ),
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) > convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) < convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) >= convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) <= convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) == convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) != convert_bool_array_to_string(y)
        ),
        getattr(UnaryOperator, "!"): lambda x: BooleanLiteral(not any(v.value for v in x.values)),
    },
}

type_hierarchy = (
    BooleanLiteral,
    IntegerLiteral,
    FloatLiteral,
    ArrayLiteral,
    SymbolLiteral,
)

builtin_constants = {
    "pi": SymbolLiteral(sympy.pi),
    "π": SymbolLiteral(sympy.pi),
    "tau": SymbolLiteral(2 * sympy.pi),
    "τ": SymbolLiteral(2 * sympy.pi),
    "euler": SymbolLiteral(sympy.E),
    "ℇ": SymbolLiteral(sympy.E),
}


class BuiltinFunctions:
    @staticmethod
    def sizeof(array: ArrayLiteral, dim: Optional[IntegerLiteral] = None) -> IntegerLiteral:
        return (
            IntegerLiteral(len(array.values))
            if dim is None or dim.value == 0
            else BuiltinFunctions.sizeof(array.values[0], IntegerLiteral(dim.value - 1))
        )

    @staticmethod
    def arccos(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.acos(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.arccos(x.value))
        )

    @staticmethod
    def arcsin(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.asin(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.arcsin(x.value))
        )

    @staticmethod
    def arctan(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.atan(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.arctan(x.value))
        )

    @staticmethod
    def ceiling(x: Union[FloatLiteral, SymbolLiteral]) -> Union[IntegerLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.ceiling(x.value))
            if isinstance(x, SymbolLiteral)
            else IntegerLiteral(math.ceil(x.value))
        )

    @staticmethod
    def cos(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.cos(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.cos(x.value))
        )

    @staticmethod
    def exp(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.exp(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.exp(x.value))
        )

    @staticmethod
    def floor(x: Union[FloatLiteral, SymbolLiteral]) -> Union[IntegerLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.floor(x.value))
            if isinstance(x, SymbolLiteral)
            else IntegerLiteral(np.floor(x.value))
        )

    @staticmethod
    def log(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.log(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.log(x.value))
        )

    @staticmethod
    def mod(
        x: Union[IntegerLiteral, FloatLiteral, SymbolLiteral],
        y: Union[IntegerLiteral, FloatLiteral, SymbolLiteral],
    ) -> Union[IntegerLiteral, FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(x.value % y.value)
            if isinstance(x, SymbolLiteral) or isinstance(y, SymbolLiteral)
            else (
                IntegerLiteral(x.value % y.value)
                if isinstance(x, IntegerLiteral) and isinstance(y, IntegerLiteral)
                else FloatLiteral(x.value % y.value)
            )
        )

    @staticmethod
    def popcount(x: Union[IntegerLiteral, BitstringLiteral, ArrayLiteral]) -> IntegerLiteral:
        # does not support symbols or expressions
        return IntegerLiteral(np.binary_repr(cast_to(UintType(), x).value).count("1"))

    @staticmethod
    def pow(
        x: Union[IntegerLiteral, FloatLiteral, SymbolLiteral],
        y: Union[IntegerLiteral, FloatLiteral, SymbolLiteral],
    ) -> Union[IntegerLiteral, FloatLiteral, SymbolLiteral]:
        # parser gets confused by pow, mistaking for quantum modifier
        return (
            SymbolLiteral(x.value**y.value)
            if isinstance(x, SymbolLiteral) or isinstance(y, SymbolLiteral)
            else (
                IntegerLiteral(x.value**y.value)
                if isinstance(x, IntegerLiteral) and isinstance(y, IntegerLiteral)
                else FloatLiteral(x.value**y.value)
            )
        )

    @staticmethod
    def rotl(x, y):
        raise NotImplementedError

    @staticmethod
    def rotr(x, y):
        raise NotImplementedError

    @staticmethod
    def sin(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.sin(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.sin(x.value))
        )

    @staticmethod
    def sqrt(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.sqrt(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.sqrt(x.value))
        )

    @staticmethod
    def tan(x: Union[FloatLiteral, SymbolLiteral]) -> Union[FloatLiteral, SymbolLiteral]:
        return (
            SymbolLiteral(sympy.tan(x.value))
            if isinstance(x, SymbolLiteral)
            else FloatLiteral(np.tan(x.value))
        )


def resolve_type_hierarchy(x: LiteralType, y: LiteralType) -> Type[LiteralType]:
    """Determine output type of expression, for example: 1 + 1.0 == 2.0"""
    return max(type(x), type(y), key=type_hierarchy.index)


def evaluate_binary_expression(
    lhs: LiteralType, rhs: LiteralType, op: BinaryOperator
) -> LiteralType:
    """Evaluate a binary expression between two literals"""
    result_type = resolve_type_hierarchy(lhs, rhs)
    func = operator_maps[result_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op.name} for {result_type.__name__}")
    return func(lhs, rhs)


def evaluate_unary_expression(expression: LiteralType, op: BinaryOperator) -> LiteralType:
    """Evaluate a unary expression on a literal"""
    expression_type = type(expression)
    func = operator_maps[expression_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op.name} for {expression_type.__name__}")
    return func(expression)


def get_operator_of_assignment_operator(assignment_operator: AssignmentOperator) -> BinaryOperator:
    """Extract the binary operator related to an assignment operator, for example: += -> +"""
    return getattr(BinaryOperator, assignment_operator.name[:-1])
