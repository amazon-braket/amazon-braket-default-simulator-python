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
from typing import List

import numpy as np


class GateOperation(ABC):
    """
    The class `GateOperation` encapsulates the unitary quantum gate operation acting
    on a set of target qubits.
    """

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """np.ndarray: The matrix representation of the operation"""

    @property
    @abstractmethod
    def targets(self) -> List[int]:
        """List[int]: The qubit indices targeted by the gate operation"""
