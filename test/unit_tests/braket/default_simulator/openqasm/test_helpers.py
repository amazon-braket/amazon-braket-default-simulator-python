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

import pytest
from sympy import Symbol

from braket.default_simulator.openqasm._helpers.casting import (
    cast_to,
    convert_string_to_bool_array,
    wrap_value_into_literal,
)
from braket.default_simulator.openqasm._helpers.functions import BuiltinFunctions
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    ArrayLiteral,
    BitstringLiteral,
    BooleanLiteral,
    FloatLiteral,
    IntegerLiteral,
    SymbolLiteral,
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


@pytest.mark.parametrize(
    "string, bool_array",
    (
        (
            BitstringLiteral(0b110, 5),
            ArrayLiteral(
                [
                    BooleanLiteral(False),
                    BooleanLiteral(False),
                    BooleanLiteral(True),
                    BooleanLiteral(True),
                    BooleanLiteral(False),
                ]
            ),
        ),
        (
            BitstringLiteral(0b110, 3),
            ArrayLiteral(
                [
                    BooleanLiteral(True),
                    BooleanLiteral(True),
                    BooleanLiteral(False),
                ]
            ),
        ),
    ),
)
def test_convert_string_to_bool_array(string, bool_array):
    assert convert_string_to_bool_array(string) == bool_array


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


def test_pow():
    assert BuiltinFunctions.pow(FloatLiteral(4), FloatLiteral(0.5)) == FloatLiteral(2)
    assert BuiltinFunctions.pow(IntegerLiteral(4), FloatLiteral(0.5)) == FloatLiteral(2)
    assert BuiltinFunctions.pow(IntegerLiteral(2), IntegerLiteral(2)) == IntegerLiteral(4)
    assert BuiltinFunctions.pow(SymbolLiteral(Symbol("x")), IntegerLiteral(2)) == SymbolLiteral(
        Symbol("x") ** 2
    )
    assert BuiltinFunctions.pow(IntegerLiteral(2), SymbolLiteral(Symbol("x"))) == SymbolLiteral(
        2 ** Symbol("x")
    )
