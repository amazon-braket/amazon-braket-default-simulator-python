import warnings
from abc import ABC
from typing import List, Union, TypeVar, Generic, Tuple

import numpy as np


class QasmType(ABC):
    data_type = None    # Name of the variable type

    __slots__ = ()

    def __eq__(self, other):
        return repr(self) == repr(other)


class QasmPrimitiveType(QasmType):

    __slots__ = ('_value', '_size')

    def __init__(self, value=None, size=None):
        self._value = value
        self._size = size
        if size is not None:
            self.validate_size(size)
        if value is not None:
            self.assign_value(value)

    supports_size = True

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return self._size

    def assign_value(self, value):
        """ Validate value is valid and assign to variable """

    def validate_size(self, size):
        """ Validate size is valid for variable """
        if not (size > 0 and size == int(size)):
            raise ValueError(
                f"{self.data_type.capitalize()} size must be a positive integer. "
                f"Provided size '{size}' for {self.data_type}."
            )

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(value={self.value}, size={self._size})"


class QubitPointer(QasmPrimitiveType):
    """ Qubit pointers """

    data_type = "qubit register"


class Bit(QasmPrimitiveType):
    """
    Single bits can be initialized with true/false, or 0/1.
    Bit registers can be initialized with a binary string of correct length.
    """

    data_type = "bit register"

    @property
    def value(self):
        if self._value is not None:
            if self._size:
                return np.binary_repr(self._value, self._size)
            else:
                return int(self._value)

    def assign_value(self, value):
        if self.size:
            try:
                assert len(value) == self.size
                self._value = int(f"0b{value}", base=2)
            except (AssertionError, ValueError, TypeError):
                raise ValueError(
                    f"Invalid value to initialize bit register: {repr(value)}. "
                    "Provided value must be a binary string of length equal to "
                    f"given size, {self.size}."
                )
        else:
            if value not in (True, False):
                raise ValueError(
                    f"Invalid value to initialize bit variable: {repr(value)}. "
                    "Provided value must be a boolean value."
                )
            self._value = bool(value)


class Int(QasmPrimitiveType):
    """
    Integers have one sign bit and (size - 1) integer bits
    Valid values are in [-2**(size - 1) + 1, 2**(size - 1) - 1]
    """

    data_type = "integer register"

    @property
    def limit(self):
        return 2 ** (self.size - 1)

    @property
    def value(self):
        if self._value is not None:
            return np.sign(self._value) * (np.abs(self._value) % self.limit)

    def assign_value(self, value):
        if isinstance(value, QasmPrimitiveType):
            value = value.value
        if int(value) != value:
            raise ValueError(
                f"Not a valid value for {self.data_type}: {repr(value)}"
            )
        self._value = value
        if self._value != self.value:
            warnings.warn(
                f"Integer overflow for {self.data_type}. "
                f"Value '{self._value}' is outside the range for an "
                f"{self.data_type} of size '{self.size}'."
            )


class Uint(Int):
    """
    Valid values are in [0, 2**size - 1]
    """

    data_type = "unsigned integer register"

    @property
    def limit(self):
        return 2 ** self.size


class Float(QasmPrimitiveType):
    """
    Floats must have a size in { 16, 32, 64, 128 }.
    """

    data_type = "float"

    def assign_value(self, value):
        try:
            self._value = np.array(value, dtype=np.dtype(f"float{self.size}"))
        except ValueError:
            raise ValueError(
                f"Not a valid value for {self.data_type}[{self.size}]: {repr(value)}"
            )

    def validate_size(self, size):
        if size not in (16, 32, 64, 128):
            raise ValueError(
                f"{self.data_type.capitalize()} size must be one of {{16, 32, 64, 128}}. "
                f"Provided size '{size}' for {self.data_type}."
            )


class Angle(QasmPrimitiveType):
    """
    Fixed point angles
    """

    data_type = "angle"

    @property
    def value(self):
        if self._value is not None:
            return (2 * np.pi * self._value) / (2 ** self.size)

    def assign_value(self, value):
        try:
            self._value = int((value % (2 * np.pi)) / (2 * np.pi) * 2 ** self.size)
        except (ValueError, TypeError):
            raise ValueError(
                f"Not a valid value for {self.data_type}: {repr(value)}"
            )


class Bool(QasmPrimitiveType):
    """
    Boolean values
    """

    supports_size = False
    data_type = "bool"

    def assign_value(self, value):
        self._value = bool(value)


class Complex(QasmPrimitiveType):
    # WIP, parser errors with complex values
    """
    Complex values with components of type Int, Uint, Float, and Angle
    """

    supports_size = False

    def __init__(
        self,
        value: Tuple[QasmPrimitiveType, QasmPrimitiveType] = None,
        base_type: QasmPrimitiveType = None,
    ):
        self._base_type = base_type
        super().__init__(value)

    @property
    def data_type(self):
        """ Name of the variable type """
        return f"complex {self._base_type.data_type}"


QasmArrayType = List[Union['QasmArrayType', QasmPrimitiveType]]


class Array(QasmPrimitiveType):
    """
    Value is an array of QasmTypes, size is a list of integers
    giving the dimensions of the array, base_type is uninitialized
    type value of some QasmType
    """

    __slots__ = ('_base_type',)

    def __init__(
        self,
        value: QasmArrayType = None,
        size: List[int] = None,
        base_type: QasmPrimitiveType = None,
    ):
        self._base_type = base_type
        super().__init__(value, size)

    def validate_size(self, size):
        if not all((s > 0 and s == int(s)) for s in size):
            raise ValueError(
                f"{self.data_type.capitalize()} dimensions must be positive integers. "
                f"Provided dimensions '{size}' for {self.data_type}."
            )

    @property
    def data_type(self):
        """ Name of the variable type """
        return f"{self._base_type.data_type} array"

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return self._size


def sample_qubit(qubit):
    p1 = np.absolute(qubit[1]) ** 2
    return np.random.binomial(1, p1)


# noinspection NonAsciiCharacters
def generate_unitary(θ, ϕ, λ):
    return np.array([
        [np.cos(θ / 2), -np.exp(1j * λ) * np.sin(θ / 2)],
        [np.exp(1j * ϕ) * np.sin(θ / 2), np.exp(ϕ + λ) * np.cos(θ / 2)],
    ])


# gate execution:
#
# use args to assign new scope of local variables (taken from gate definition)
# do the same for targets (in the same scope)
#
# execute the gate, manipulating local variables
# initially this step can use the ast code
# in the future, we may only want to parse this once
#
# pop the scope

class Gate(QasmType):

    data_type = "gate definition"

    __slots__ = ('_params', '_targets', '_body')

    def __init__(self, params=(), targets=(), body=()):
        self._params = params
        self._targets = targets
        self._body = body

    @property
    def params(self):
        return self._params

    @property
    def targets(self):
        return self._targets

    @property
    def body(self):
        return self._body

    def __repr__(self):
        return f"Gate(params={self._params}, targets={self._targets}, body=[...])"


class GateCall(QasmType):

    data_type = "gate call"

    __slots__ = ('_name', '_params', '_targets', '_modifiers')

    def __init__(self, name, params=(), targets=(), modifiers=()):
        self._name = name
        self._params = params
        self._targets = targets
        self._modifiers = modifiers

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params

    @property
    def targets(self):
        return self._targets

    @property
    def modifiers(self):
        return self._modifiers

    def __repr__(self):
        return (
            f"GateCall({self._name}, params={self._params}, "
            f"targets={self._targets}, mods={self._modifiers})"
        )


class Number(QasmPrimitiveType):

    data_type = "number"
