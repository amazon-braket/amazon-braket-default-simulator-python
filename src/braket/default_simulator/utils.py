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

from functools import lru_cache

import numpy as np


@lru_cache()
def pauli_eigenvalues(n: int) -> np.ndarray:
    """ The eigenvalues of Pauli operators and their tensor products.

    Args:
        n (int): the number of qubits the operator acts on
    Returns:
        np.ndarray: the eigenvalues of a Pauli product operator of the given size
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigenvalues(n - 1), -pauli_eigenvalues(n - 1)])
