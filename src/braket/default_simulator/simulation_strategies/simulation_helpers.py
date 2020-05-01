# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from functools import singledispatch

from braket.default_simulator.operation import GateOperation, Observable


@singledispatch
def get_matrix(operation):
    """ Gets the matrix of the given operation.

    For a `GateOperation`, this is the gate's unitary matrix, and for an `Observable`,
    this is its diagonalizing matrix.

    Args:
        operation: The operation whose matrix is needed

    Returns:
        np.ndarray: The matrix of the operation
    """
    raise ValueError(f"Unrecognized operation: {operation}")


@get_matrix.register
def _(gate: GateOperation):
    return gate.matrix


@get_matrix.register
def _(observable: Observable):
    return observable.diagonalizing_matrix
