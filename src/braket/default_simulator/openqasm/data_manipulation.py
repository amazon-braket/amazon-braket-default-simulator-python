import warnings
from functools import singledispatchmethod
from typing import Union

import numpy as np
from openqasm.ast import (
    BinaryOperator,
    BitType,
    BooleanLiteral,
    ClassicalType,
    Constant,
    ConstantName,
    Expression,
    FloatType,
    IntegerLiteral,
    IntType,
    RealLiteral,
    StringLiteral,
    UintType,
    UnaryOperator,
)

LiteralType = Union[BooleanLiteral, IntegerLiteral, RealLiteral]


class DataManipulation:

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

    @classmethod
    def _returns_boolean(cls, op: BinaryOperator):
        return op in (
            getattr(BinaryOperator, ">"),
            getattr(BinaryOperator, "<"),
            getattr(BinaryOperator, ">="),
            getattr(BinaryOperator, "<="),
            getattr(BinaryOperator, "=="),
            getattr(BinaryOperator, "!="),
        )

    @classmethod
    def resolve_result_type(
        cls, x: Union[ClassicalType, LiteralType], y: Union[ClassicalType, LiteralType]
    ):
        # TODO: add support for actual ClassicalTypes, not just literals
        return max(x, y, key=cls.type_hierarchy.index)

    @singledispatchmethod
    @classmethod
    def cast_to(cls, into: Union[ClassicalType, LiteralType], variable: LiteralType):
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
    @classmethod
    def _(cls, into: BitType, variable: LiteralType):
        if not into.size:
            return cls.cast_to(BooleanLiteral, variable)
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
            return cls.cast_to(IntType(into.size), variable)

    @cast_to.register
    @classmethod
    def _(cls, into: IntType, variable: LiteralType):
        limit = 2 ** (into.size.value - 1)
        value = np.sign(variable.value) * (np.abs(variable.value) % limit)
        if value != variable.value:
            warnings.warn(
                f"Integer overflow for value {variable.value} and size {into.size.value}."
            )
        return IntegerLiteral(value)

    @cast_to.register
    @classmethod
    def _(cls, into: UintType, variable: LiteralType):
        limit = 2 ** into.size.value
        value = variable.value % limit
        if value != variable.value:
            warnings.warn(
                f"Unsigned integer overflow for value {variable.value} and size {into.size.value}."
            )
        return IntegerLiteral(value)

    @cast_to.register
    @classmethod
    def _(cls, into: FloatType, variable: LiteralType):
        if into.size.value not in (16, 32, 64, 128):
            raise ValueError("Float size must be one of {{16, 32, 64, 128}}.")
        value = float(np.array(variable.value, dtype=np.dtype(f"float{into.size.value}")))
        return RealLiteral(value)

    @classmethod
    def evaluate_binary_expression(cls, lhs: Expression, rhs: Expression, op: BinaryOperator):
        # assume lhs and rhs are of same type
        result_type = type(lhs)
        func = cls.operator_maps[result_type].get(op)
        if not func:
            raise TypeError(f"Invalid operator {op} for {result_type.__name__}")
        return_type = BooleanLiteral if cls._returns_boolean(op) else result_type
        return return_type(func(lhs.value, rhs.value))

    @classmethod
    def evaluate_unary_expression(cls, expression: Expression, op: BinaryOperator):
        # assume lhs and rhs are of same type
        result_type = type(expression)
        func = cls.operator_maps[result_type].get(op)
        if not func:
            raise TypeError(f"Invalid operator {op} for {result_type.__name__}")
        return_type = BooleanLiteral if cls._returns_boolean(op) else result_type
        return return_type(func(expression.value))

    @classmethod
    def evaluate_constant(cls, constant: Constant):
        return RealLiteral(cls.constant_map.get(constant.name))
