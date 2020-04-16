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

import itertools
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
from braket.default_simulator.operation import Observable
from braket.default_simulator.operation_helpers import (
    check_hermitian,
    check_matrix_dimensions,
    pauli_eigenvalues,
)


class Identity(Observable):
    """Identity observable

    Note:
        This observable refers to the same mathematical object as the gate operation
        of the same name, but is meant to be used differently; the observable is viewed
        as a Hermitian operator to be measured, while the gate is viewed as a unitary
        operator to evolve the state of the system.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        self._targets = _validate_and_clone_single_qubit_target(targets)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return False

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.array([1, 1])

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        return None


class Hadamard(Observable):
    """Hadamard observable

    Note:
        This observable refers to the same mathematical object as the gate operation
        of the same name, but is meant to be used differently; the observable is viewed
        as a Hermitian operator to be measured, while the gate is viewed as a unitary
        operator to evolve the state of the system.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        self._targets = _validate_and_clone_single_qubit_target(targets)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return True

    @property
    def eigenvalues(self) -> np.ndarray:
        return pauli_eigenvalues(1)

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        # RY(-\pi / 4)
        angle = -np.pi / 4
        cos_component = np.cos(angle / 2)
        sin_component = np.sin(angle / 2)
        return np.array([[cos_component, -sin_component], [sin_component, cos_component]])


class PauliX(Observable):
    """Pauli-X observable

    Note:
        This observable refers to the same mathematical object as the gate operation
        of the same name, but is meant to be used differently; the observable is viewed
        as a Hermitian operator to be measured, while the gate is viewed as a unitary
        operator to evolve the state of the system.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        self._targets = _validate_and_clone_single_qubit_target(targets)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return True

    @property
    def eigenvalues(self) -> np.ndarray:
        return pauli_eigenvalues(1)

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        # H
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class PauliY(Observable):
    """Pauli-Y observable

    Note:
        This observable refers to the same mathematical object as the gate operation
        of the same name, but is meant to be used differently; the observable is viewed
        as a Hermitian operator to be measured, while the gate is viewed as a unitary
        operator to evolve the state of the system.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        self._targets = _validate_and_clone_single_qubit_target(targets)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return True

    @property
    def eigenvalues(self) -> np.ndarray:
        return pauli_eigenvalues(1)

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        # HS^{\dagger}
        return np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)


class PauliZ(Observable):
    """Pauli-Z observable

    Note:
        This observable refers to the same mathematical object as the gate operation
        of the same name, but is meant to be used differently; the observable is viewed
        as a Hermitian operator to be measured, while the gate is viewed as a unitary
        operator to evolve the state of the system.
    """

    def __init__(self, targets: Optional[List[int]] = None):
        self._targets = _validate_and_clone_single_qubit_target(targets)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return True

    @property
    def eigenvalues(self) -> np.ndarray:
        return pauli_eigenvalues(1)

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        return None


class Hermitian(Observable):
    """Arbitrary Hermitian observable"""

    # Cache of eigenpairs for each used Hermitian matrix
    _eigenpairs = {}

    def __init__(self, matrix: np.ndarray, targets: Optional[List[int]] = None):
        clone = np.array(matrix, dtype=complex)
        self._targets = tuple(targets) if targets else None
        if targets:
            check_matrix_dimensions(clone, self._targets)
        elif clone.shape != (2, 2):
            raise ValueError(
                f"Matrix must have shape (2, 2) if target is empty, but has shape {clone.shape}"
            )
        check_hermitian(clone)
        self._matrix = clone

    @property
    def matrix(self) -> np.ndarray:
        """np.ndarray: The Hermitian matrix defining the observable."""
        return np.array(self._matrix)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return False

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigendecomposition()["eigenvalues"]

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        return self._eigendecomposition()["eigenvectors"].conj().T

    def _eigendecomposition(self) -> Dict[str, np.ndarray]:
        """ Decomposes the Hermitian matrix into its eigenvectors and associated eigenvalues.

        The eigendecomposition is cached so that if another Hermitian observable
        is created with the same matrix, the eigendecomposition doesn't have to
        be recalculated.

        Returns:
            Dict[str, np.ndarray]: The keys are "eigenvectors", mapping to a matrix whose
            columns are the eigenvectors of the matrix, and "eigenvalues", a list of
            associated eigenvalues in the order their corresponding eigenvectors in the
            "eigenvectors" matrix
        """
        mat_key = tuple(self._matrix.flatten().tolist())
        if mat_key not in Hermitian._eigenpairs:
            eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
            Hermitian._eigenpairs[mat_key] = {
                "eigenvectors": eigenvectors,
                "eigenvalues": eigenvalues,
            }
        return Hermitian._eigenpairs[mat_key]


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
        if len(constituents) < 2:
            raise ValueError("A tensor product should have at least 2 constituent observables")
        self._targets = tuple(
            target for observable in constituents for target in observable.targets
        )
        self._diagonalizing_matrix = TensorProduct._construct_matrix(constituents)
        self._eigenvalues = TensorProduct._compute_eigenvalues(constituents, self._targets)

    @property
    def targets(self) -> Tuple[int]:
        return self._targets

    @property
    def is_standard(self) -> bool:
        return False

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @property
    def diagonalizing_matrix(self) -> Optional[np.ndarray]:
        return self._diagonalizing_matrix

    @staticmethod
    def _construct_matrix(constituents: List[Observable]) -> np.ndarray:
        return reduce(
            # (A \otimes I)(I \otimes B) == A \otimes B
            lambda a, b: np.kron(a, b),
            [
                observable.diagonalizing_matrix
                for observable in constituents
                # Ignore observables with trivial diagonalizing matrices
                if observable.diagonalizing_matrix is not None
            ],
        )

    @staticmethod
    def _compute_eigenvalues(constituents: List[Observable], targets: Tuple[int]) -> np.ndarray:
        # check if there are any non-standard observables, such as Hermitian
        if False in {observable.is_standard for observable in constituents}:
            obs_sorted = sorted(constituents, key=lambda x: x.targets)

            # Tensor product of observables contains a mixture
            # of standard and non-standard observables
            eigenvalues = np.array([1])
            for k, g in itertools.groupby(obs_sorted, lambda x: x.is_standard):
                if k:
                    # Subgroup g contains only standard observables.
                    eigenvalues = np.kron(eigenvalues, pauli_eigenvalues(len(list(g))))
                else:
                    # Subgroup g contains only non-standard observables.
                    for nonstandard in g:
                        # loop through all non-standard observables
                        eigenvalues = np.kron(eigenvalues, nonstandard.eigenvalues)
        else:
            eigenvalues = pauli_eigenvalues(len(targets))

        return eigenvalues


def _validate_and_clone_single_qubit_target(targets: Optional[List[int]]) -> Optional[Tuple[int]]:
    clone = tuple(targets) if targets else None
    if clone and len(clone) > 1:
        raise ValueError(f"Observable only acts on one qubit, but found {len(clone)}")
    return clone
