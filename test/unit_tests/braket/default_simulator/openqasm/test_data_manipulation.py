import pytest
from openqasm3.ast import (
    BooleanLiteral,
    DiscreteSet,
    Identifier,
    IndexedIdentifier,
    IntegerLiteral,
    RealLiteral,
)

from braket.default_simulator.openqasm.data_manipulation import (
    cast_to,
    convert_to_output,
    get_identifier_string,
    wrap_value_into_literal,
)


def test_cast():
    assert cast_to(IntegerLiteral, RealLiteral(1.5)) == IntegerLiteral(1)
    assert cast_to(RealLiteral, IntegerLiteral(5)) == RealLiteral(5)


def test_undefined_cast():
    class UndefinedType:
        pass

    cannot_cast = "Cannot cast IntegerLiteral into UndefinedType."
    with pytest.raises(TypeError, match=cannot_cast):
        cast_to(UndefinedType, IntegerLiteral(1))


def test_nested_indices_print():
    with pytest.raises(NotImplementedError, match="nested indices string conversion"):
        get_identifier_string(
            IndexedIdentifier(
                Identifier("q"),
                [[IntegerLiteral(0)], [IntegerLiteral(1)]],
            )
        )

    with pytest.raises(NotImplementedError, match="nested indices string conversion"):
        get_identifier_string(
            IndexedIdentifier(
                Identifier("q"),
                [[IntegerLiteral(0), IntegerLiteral(1)]],
            )
        )

    with pytest.raises(NotImplementedError, match="Discrete set indexed identifier string"):
        get_identifier_string(
            IndexedIdentifier(
                Identifier("q"),
                [DiscreteSet([IntegerLiteral(0), IntegerLiteral(1)])],
            )
        )


def test_undefined_output():
    cannot_convert = "converting {'a': 1} to output"
    with pytest.raises(TypeError, match=cannot_convert):
        convert_to_output({"a": 1})


def test_undefined_wrap():
    cannot_wrap = "Cannot wrap {'a': 1} into literal type"
    with pytest.raises(TypeError, match=cannot_wrap):
        wrap_value_into_literal({"a": 1})


def test_wrap_bool():
    assert wrap_value_into_literal(True) == BooleanLiteral(True)
