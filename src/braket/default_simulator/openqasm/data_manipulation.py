import warnings
from functools import singledispatch
from typing import Union

import numpy as np
from openqasm3.ast import (
    ArrayLiteral,
    ArrayType,
    BinaryOperator,
    BitType,
    BooleanLiteral,
    ClassicalType,
    Constant,
    ConstantName,
    DiscreteSet,
    Expression,
    FloatType,
    IndexElement,
    IntegerLiteral,
    IntType,
    RealLiteral,
    StringLiteral,
    UintType,
    UnaryOperator,
)

LiteralType = Union[BooleanLiteral, IntegerLiteral, RealLiteral]

operator_maps = {
    IntegerLiteral: {
        # returns int
        getattr(BinaryOperator, "+"): lambda x, y: x + y,
        getattr(BinaryOperator, "-"): lambda x, y: x - y,
        getattr(BinaryOperator, "*"): lambda x, y: x * y,
        getattr(BinaryOperator, "/"): lambda x, y: x // y,
        getattr(BinaryOperator, "%"): lambda x, y: x % y,
        getattr(BinaryOperator, "**"): lambda x, y: x ** y,
        getattr(UnaryOperator, "-"): lambda x: -x,
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: x > y,
        getattr(BinaryOperator, "<"): lambda x, y: x < y,
        getattr(BinaryOperator, ">="): lambda x, y: x >= y,
        getattr(BinaryOperator, "<="): lambda x, y: x <= y,
        getattr(BinaryOperator, "=="): lambda x, y: x == y,
        getattr(BinaryOperator, "!="): lambda x, y: x != y,
    },
    RealLiteral: {
        # returns real
        getattr(BinaryOperator, "+"): lambda x, y: x + y,
        getattr(BinaryOperator, "-"): lambda x, y: x - y,
        getattr(BinaryOperator, "*"): lambda x, y: x * y,
        getattr(BinaryOperator, "/"): lambda x, y: x / y,
        getattr(BinaryOperator, "%"): lambda x, y: x % y,
        getattr(BinaryOperator, "**"): lambda x, y: x ** y,
        getattr(UnaryOperator, "-"): lambda x: -x,
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: x > y,
        getattr(BinaryOperator, "<"): lambda x, y: x < y,
        getattr(BinaryOperator, ">="): lambda x, y: x >= y,
        getattr(BinaryOperator, "<="): lambda x, y: x <= y,
        getattr(BinaryOperator, "=="): lambda x, y: x == y,
        getattr(BinaryOperator, "!="): lambda x, y: x != y,
    },
    BooleanLiteral: {
        # returns bool
        getattr(BinaryOperator, "&&"): lambda x, y: x and y,
        getattr(BinaryOperator, "||"): lambda x, y: x or y,
        getattr(BinaryOperator, ">"): lambda x, y: x > y,
        getattr(BinaryOperator, "<"): lambda x, y: x < y,
        getattr(BinaryOperator, ">="): lambda x, y: x >= y,
        getattr(BinaryOperator, "<="): lambda x, y: x <= y,
        getattr(BinaryOperator, "=="): lambda x, y: x == y,
        getattr(BinaryOperator, "!="): lambda x, y: x != y,
        getattr(UnaryOperator, "!"): lambda x: not x,
    },
    # comprehensive list for ref (will delete)
    getattr(BinaryOperator, ">"): lambda x, y: x > y,
    getattr(BinaryOperator, "<"): lambda x, y: x < y,
    getattr(BinaryOperator, ">="): lambda x, y: x >= y,
    getattr(BinaryOperator, "<="): lambda x, y: x <= y,
    getattr(BinaryOperator, "=="): lambda x, y: x == y,
    getattr(BinaryOperator, "!="): lambda x, y: x != y,
    getattr(BinaryOperator, "&&"): lambda x, y: x and y,
    getattr(BinaryOperator, "||"): lambda x, y: x or y,
    getattr(BinaryOperator, "|"): lambda x, y: x | y,
    getattr(BinaryOperator, "^"): lambda x, y: x ^ y,
    getattr(BinaryOperator, "&"): lambda x, y: x & y,
    getattr(BinaryOperator, "<<"): lambda x, y: x << y,
    getattr(BinaryOperator, ">>"): lambda x, y: x >> y,
    getattr(BinaryOperator, "+"): lambda x, y: x + y,
    getattr(BinaryOperator, "-"): lambda x, y: x - y,
    getattr(BinaryOperator, "*"): lambda x, y: x * y,
    getattr(BinaryOperator, "/"): lambda x, y: x / y,
    getattr(BinaryOperator, "%"): lambda x, y: x % y,
    getattr(BinaryOperator, "**"): lambda x, y: x ** y,
}

type_hierarchy = [
    BooleanLiteral,
    IntegerLiteral,
    RealLiteral,
]

constant_map = {
    ConstantName.pi: np.pi,
    ConstantName.tau: 2 * np.pi,
    ConstantName.euler: np.e,
}


def _returns_boolean(op: BinaryOperator):
    return op in (
        getattr(BinaryOperator, ">"),
        getattr(BinaryOperator, "<"),
        getattr(BinaryOperator, ">="),
        getattr(BinaryOperator, "<="),
        getattr(BinaryOperator, "=="),
        getattr(BinaryOperator, "!="),
    )


def resolve_result_type(x: Union[ClassicalType, LiteralType], y: Union[ClassicalType, LiteralType]):
    # TODO: add support for actual ClassicalTypes, not just literals
    return max(x, y, key=type_hierarchy.index)


@singledispatch
def cast_to(into: Union[ClassicalType, LiteralType], variable: LiteralType):
    if type(variable) == into:
        return variable
    if into == BooleanLiteral:
        return BooleanLiteral(bool(variable.value))
    if into == IntegerLiteral:
        return IntegerLiteral(int(variable.value))
    if into == RealLiteral:
        return RealLiteral(float(variable.value))
    raise TypeError(f"Cannot cast {type(variable)} into {into}.")


@cast_to.register
def _(into: BitType, variable: LiteralType):
    if not into.size:
        return cast_to(BooleanLiteral, variable)
    else:
        if isinstance(variable, StringLiteral):
            try:
                assert len(variable.value) == into.size.value
                variable = IntegerLiteral(int(f"0b{variable.value}", base=2))
            except (AssertionError, ValueError, TypeError):
                raise ValueError(
                    f"Invalid string to initialize bit register of size {into.size.value}: "
                    f"'{variable.value}'"
                )
        return cast_to(UintType(into.size), variable)


@cast_to.register
def _(into: IntType, variable: LiteralType):
    limit = 2 ** (into.size.value - 1)
    value = int(np.sign(variable.value) * (np.abs(int(variable.value)) % limit))
    if value != variable.value:
        warnings.warn(f"Integer overflow for value {variable.value} and size {into.size.value}.")
    return IntegerLiteral(value)


@cast_to.register
def _(into: UintType, variable: LiteralType):
    limit = 2 ** into.size.value
    value = int(variable.value) % limit
    if value != variable.value:
        warnings.warn(
            f"Unsigned integer overflow for value {variable.value} and size {into.size.value}."
        )
    return IntegerLiteral(value)


@cast_to.register
def _(into: FloatType, variable: LiteralType):
    if into.size.value not in (16, 32, 64, 128):
        raise ValueError("Float size must be one of {{16, 32, 64, 128}}.")
    value = float(np.array(variable.value, dtype=np.dtype(f"float{into.size.value}")))
    return RealLiteral(value)


@cast_to.register
def _(into: ArrayType, variable: Union[ArrayLiteral, DiscreteSet]):
    if len(variable.values) != into.dimensions[0].value:
        raise ValueError(
            f"Size mismatch between dimension of size {into.dimensions[0].value} "
            f"and values length {len(variable.values)}"
        )
    subtype = (
        ArrayType(into.base_type, into.dimensions[1:])
        if len(into.dimensions) > 1
        else into.base_type
    )
    return ArrayLiteral([cast_to(subtype, v) for v in variable.values])


def evaluate_binary_expression(lhs: Expression, rhs: Expression, op: BinaryOperator):
    # assume lhs and rhs are of same type
    result_type = type(lhs)
    func = operator_maps[result_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op} for {result_type.__name__}")
    return_type = BooleanLiteral if _returns_boolean(op) else result_type
    return return_type(func(lhs.value, rhs.value))


def evaluate_unary_expression(expression: Expression, op: BinaryOperator):
    # assume lhs and rhs are of same type
    result_type = type(expression)
    func = operator_maps[result_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op} for {result_type.__name__}")
    return_type = BooleanLiteral if _returns_boolean(op) else result_type
    return return_type(func(expression.value))


def evaluate_constant(constant: Constant):
    return RealLiteral(constant_map.get(constant.name))


def get_elements(array: ArrayLiteral, index: IndexElement):
    if isinstance(index, DiscreteSet):
        return DiscreteSet([get_elements(array, [i]) for i in index.values])
    if not isinstance(index, list):
        index = [index]
    for i in index:
        array = array.values[i.value]
    return array
