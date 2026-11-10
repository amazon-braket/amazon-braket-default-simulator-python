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

from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.default_simulator.openqasm.parser import openqasm_ast as ast
from braket.default_simulator.openqasm.parser.openqasm_parser import parse


@pytest.mark.parametrize("access", ["readonly", "const", "mutable"])
@pytest.mark.parametrize(
    "dimensions, expected_dimensions",
    [
        ("#dim = 1", ast.IntegerLiteral(1)),
        ("2, 3", [ast.IntegerLiteral(2), ast.IntegerLiteral(3)]),
    ],
)
@pytest.mark.parametrize("declaration", ["def", "extern"])
def test_array_reference_parameter_ast(access, dimensions, expected_dimensions, declaration):
    if declaration == "def":
        source = f"def read({access} array[int[32], {dimensions}] values) {{}}"
    else:
        source = f"extern read({access} array[int[32], {dimensions}]);"

    program = parse(f"OPENQASM 3.0; {source}")
    argument = program.statements[0].arguments[0]

    expected_access = ast.AccessControl.mutable if access == "mutable" else ast.AccessControl.const
    assert argument.access == expected_access
    assert argument.type == ast.ArrayReferenceType(
        base_type=ast.IntType(ast.IntegerLiteral(32)), dimensions=expected_dimensions
    )
    if declaration == "def":
        assert argument.name == ast.Identifier("values")


@pytest.mark.parametrize("access", ["readonly", "const", "mutable"])
@pytest.mark.parametrize("dimensions", ["#dim = 1", "2"])
def test_array_reference_parameter_read(access, dimensions):
    source = f"""
    OPENQASM 3.0;
    def first({access} array[int[32], {dimensions}] values) -> int[32] {{
        return values[0];
    }}
    array[int[32], 2] values = {{5, 6}};
    int[32] result = first(values);
    """

    context = Interpreter().run(source)

    assert context.get_value("result") == ast.IntegerLiteral(5)
    assert context.get_value("values") == ast.ArrayLiteral(
        [ast.IntegerLiteral(5), ast.IntegerLiteral(6)]
    )


@pytest.mark.parametrize("access", ["readonly", "const"])
@pytest.mark.parametrize("dimensions", ["#dim = 1", "2"])
@pytest.mark.parametrize("assignment", ["values[0] = 7;", "values[0] += 1;"])
def test_readonly_array_reference_rejects_assignment(access, dimensions, assignment):
    source = f"""
    OPENQASM 3.0;
    def change({access} array[int[32], {dimensions}] values) {{
        {assignment}
    }}
    array[int[32], 2] values = {{5, 6}};
    change(values);
    """

    with pytest.raises(TypeError, match="Cannot update const value values"):
        Interpreter().run(source)


@pytest.mark.parametrize("dimensions", ["#dim = 1", "2"])
def test_mutable_array_reference_updates_caller(dimensions):
    source = f"""
    OPENQASM 3.0;
    def change(mutable array[int[32], {dimensions}] values) {{
        values[0] = 7;
    }}
    array[int[32], 2] data = {{5, 6}};
    change(data);
    """

    context = Interpreter().run(source)

    assert context.get_value("data") == ast.ArrayLiteral(
        [ast.IntegerLiteral(7), ast.IntegerLiteral(6)]
    )


def test_readonly_keyword_does_not_split_identifiers():
    source = """
    OPENQASM 3.0;
    def readonly_sum(int readonly_value, int readonly2) -> int {
        return readonly_value + readonly2;
    }
    int result = readonly_sum(2, 3);
    """

    context = Interpreter().run(source)

    assert context.get_value("result") == ast.IntegerLiteral(5)
