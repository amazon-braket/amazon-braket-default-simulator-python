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

import itertools
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional

import numpy as np
from braket.default_simulator.operation_helpers import pauli_eigenvalues


class GateOperation(ABC):
    """
    Encapsulates a unitary quantum gate operation acting on
    a set of target qubits.
    """

    @property
    @abstractmethod
    def targets(self) -> List[int]:
        """List[int]: The target qubit indices of the operation."""

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """np.ndarray: The matrix representation of the operation."""


class Observable(ABC):
    """
    Encapsulates an observable to be measured in the computational basis.
    """

    @property
    @abstractmethod
    def targets(self) -> List[int]:
        """List[int]: The target qubit indices of the operation."""

    @property
    @abstractmethod
    def is_standard(self) -> bool:
        """ bool: Whether the observable is one of the four standard observables.

        Namely, X, Y, Z or H; these observables are guaranteed to have eigenvalues
        of +/-1
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


class TensorProduct(Observable):
    """
    Tensor product of multiple observables.
    """

    def __init__(self, constituents: List[Observable]):
        """
        Args:
            constituents (List[Observable]): The observables being combined together
                via tensor product
        """
        self._constituents = constituents
        self._targets = [target for observable in constituents for target in observable.targets]
        self._eigenvalues = None

    @property
    def targets(self) -> List[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return False

    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is not None:
            return self._eigenvalues

        # check if there are any non-standard observables, such as Hermitian
        if False in {observable.is_standard for observable in self._constituents}:
            obs_sorted = sorted(self._constituents, key=lambda x: x.targets)

            # Tensor product of observables contains a mixture
            # of standard and non-standard observables
            self._eigenvalues = np.array([1])
            for k, g in itertools.groupby(obs_sorted, lambda x: x.is_standard):
                if k:
                    # Subgroup g contains only standard observables.
                    self._eigenvalues = np.kron(self._eigenvalues, pauli_eigenvalues(len(list(g))))
                else:
                    # Subgroup g contains only non-standard observables.
                    for nonstandard in g:
                        # loop through all non-standard observables
                        self._eigenvalues = np.kron(self._eigenvalues, nonstandard.eigenvalues)
        else:
            self._eigenvalues = pauli_eigenvalues(len(self.targets))

        return self._eigenvalues

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        return reduce(
            lambda a, b: np.tensordot(a, b, 0),
            # Ignore observables with trivial diagonalizing matrices
            [
                observable.diagonalizing_matrix
                for observable in self._constituents
                if observable.diagonalizing_matrix is not None
            ],
        )
