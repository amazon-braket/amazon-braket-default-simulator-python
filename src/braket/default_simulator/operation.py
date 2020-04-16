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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Operation(ABC):
    """
    Encapsulates an operation acting on a set of target qubits.
    """

    @property
    @abstractmethod
    def targets(self) -> Tuple[int]:
        """Tuple[int]: The target qubit indices of the operation."""


class GateOperation(Operation, ABC):
    """
    Encapsulates a unitary quantum gate operation acting on
    a set of target qubits.
    """

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """np.ndarray: The matrix representation of the operation."""


class Observable(Operation, ABC):
    """
    Encapsulates an observable to be measured in the computational basis.
    """

    @property
    @abstractmethod
    def is_standard(self) -> bool:
        """ bool: Whether the observable is one of the four standard observables X, Y, Z and H.

        These observables are guaranteed to have eigenvalues of +/-1
        """

    @property
    @abstractmethod
    def eigenvalues(self) -> np.ndarray:
        """
        np.ndarray: The eigenvalues of the observable ordered by computational basis state.
        """

    @property
    @abstractmethod
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        """
        Optional[np.ndarray]: The matrix that diagonalizes the observable
        in the computational basis if it is not already in the computational basis.
        """
