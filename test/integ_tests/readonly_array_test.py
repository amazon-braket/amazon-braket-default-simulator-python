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

from braket.devices import LocalSimulator
from braket.ir.openqasm import Program


@pytest.mark.parametrize("backend", ["braket_sv", "braket_dm"])
def test_readonly_array_observes_preceding_mutable_call(backend):
    source = """
    OPENQASM 3.0;
    def change(mutable array[int, #dim = 1] values) -> int {
        values[0] = 5;
        return 0;
    }
    def read(readonly array[int, #dim = 1] values, int ignored) -> int {
        return values[0];
    }
    array[int, 1] data = {0};
    int change_result;
    output int result;
    change_result = change(data);
    result = read(data, change_result);
    """

    result = LocalSimulator(backend).run(Program(source=source), shots=4).result()

    assert result.outputs == [{"result": 5}] * 4


@pytest.mark.parametrize("backend", ["braket_sv", "braket_dm"])
@pytest.mark.parametrize("access", ["readonly", "const"])
@pytest.mark.parametrize("dimensions", ["#dim = 1", "1"])
def test_readonly_array_rejects_write(backend, access, dimensions):
    source = f"""
    OPENQASM 3.0;
    def change({access} array[int, {dimensions}] values) {{
        values[0] = 7;
    }}
    array[int, 1] data = {{5}};
    change(data);
    """

    with pytest.raises(TypeError, match="Cannot update const value values"):
        LocalSimulator(backend).run(Program(source=source), shots=4).result()
