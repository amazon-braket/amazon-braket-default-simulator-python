import pytest
from openqasm3.ast import (
    ArrayLiteral,
    BooleanLiteral,
    FloatLiteral,
    IndexExpression,
    IntegerLiteral,
)

from braket.default_simulator.openqasm.data_manipulation import (
    cast_to,
    index_expression_to_indexed_identifier,
    wrap_value_into_literal,
)


def test_cast():
    assert cast_to(IntegerLiteral, FloatLiteral(1.5)) == IntegerLiteral(1)
    assert cast_to(FloatLiteral, IntegerLiteral(5)) == FloatLiteral(5)


def test_undefined_cast():
    class UndefinedType:
        pass

    cannot_cast = "Cannot cast IntegerLiteral into UndefinedType."
    with pytest.raises(TypeError, match=cannot_cast):
        cast_to(UndefinedType, IntegerLiteral(1))


def test_undefined_wrap():
    cannot_wrap = "Cannot wrap {'a': 1} into literal type"
    with pytest.raises(TypeError, match=cannot_wrap):
        wrap_value_into_literal({"a": 1})


def test_wrap_bool():
    assert wrap_value_into_literal(True) == BooleanLiteral(True)


def test_wrap_float():
    assert wrap_value_into_literal(3.14) == FloatLiteral(3.14)


def test_wrap_array():
    assert wrap_value_into_literal([1, 2, 3]) == ArrayLiteral(
        [IntegerLiteral(1), IntegerLiteral(2), IntegerLiteral(3)]
    )
    assert wrap_value_into_literal([]) == ArrayLiteral([])


def test_convert_index_expression_wrong_type():
    index_expression = IndexExpression(ArrayLiteral([IntegerLiteral(1)]), [IntegerLiteral(0)])
    wrong_type = "Can only transform index expressions of an identifier into an IndexedIdentifier"
    with pytest.raises(TypeError, match=wrong_type):
        index_expression_to_indexed_identifier(index_expression)
