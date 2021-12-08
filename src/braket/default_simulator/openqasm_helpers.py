import warnings
from abc import ABC, abstractmethod

import numpy as np


class QasmVariable(ABC):

    def __init__(self, name, value=None, size=None):
        self.name = name
        self._value = value
        self._size = size
        if size is not None:
            self.validate_size(size)
        if value is not None:
            self.validate_value(value, size)

    @property
    @abstractmethod
    def data_type(self):
        """ Name of the variable type """

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return self._size

    def validate_value(self, value, size):
        """ Validate value is valid for variable """

    def validate_size(self, size):
        """ Validate size is valid for variable """
        if not (size > 0 and size == int(size)):
            raise ValueError(
                f"{self.data_type.capitalize()} size must be a positive integer. "
                f"Provided size '{size}' for {self.data_type} '{self.name}'."
            )

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name}, value={self.value}, size={self._size})"

    def __eq__(self, other):
        return repr(self) == repr(other)


class QubitPointer(QasmVariable):
    """ Qubit pointers """

    @property
    def data_type(self):
        return "qubit register"


class BitVariable(QasmVariable):
    """
    Single bits can be initialized with true/false, or 0/1.
    Bit registers can be initialized with a binary string of correct length.
    """

    @property
    def data_type(self):
        return "bit register"

    @property
    def value(self):
        if self._value is not None:
            if self._size:
                return np.binary_repr(self._value, self._size)
            else:
                return int(self._value)

    def validate_value(self, value, size):
        if size:
            try:
                assert len(value) == size
                self._value = int(f"0b{value}", base=2)
            except (AssertionError, ValueError, TypeError):
                raise ValueError(
                    f"Invalid value to initialize bit register '{self.name}': {repr(value)}. "
                    "Provided value must be a binary string of length equal to "
                    f"given size, {size}."
                )
        else:
            if value not in (True, False):
                raise ValueError(
                    f"Invalid value to initialize bit variable '{self.name}': {repr(value)}. "
                    "Provided value must be a boolean value."
                )
            self._value = bool(value)


class IntVariable(QasmVariable):
    """
    Integers have one sign bit and (size - 1) integer bits
    Valid values are in [-2**(size - 1) + 1, 2**(size - 1) - 1]
    """

    @property
    def limit(self):
        return 2 ** (self.size - 1)

    @property
    def data_type(self):
        return "integer register"

    @property
    def value(self):
        if self._value is not None:
            return np.sign(self._value) * (np.abs(self._value) % self.limit)

    def validate_value(self, value, size):
        if int(value) != value:
            raise ValueError(
                f"Not a valid value for {self.data_type} '{self.name}': {repr(value)}"
            )
        self._value = value
        if self._value != self.value:
            warnings.warn(
                f"Integer overflow for {self.data_type} '{self.name}'. "
                f"Value '{self._value}' is outside the range for an "
                f"{self.data_type} of size '{self.size}'."
            )


class UintVariable(IntVariable):
    """
    Valid values are in [0, 2**size - 1]
    """

    @property
    def limit(self):
        return 2 ** self.size

    @property
    def data_type(self):
        return "unsigned integer register"


class FloatVariable(QasmVariable):
    """
    Floats must have a size in { 16, 32, 64, 128 }.
    """

    @property
    def data_type(self):
        return "float"

    def validate_value(self, value, size):
        try:
            self._value = np.array(value, dtype=np.dtype(f"float{size}"))
        except ValueError:
            raise ValueError(
                f"Not a valid value for {self.data_type} '{self.name}': {repr(value)}"
            )

    def validate_size(self, size):
        if size not in (16, 32, 64, 128):
            raise ValueError(
                f"{self.data_type.capitalize()} size must be one of {{16, 32, 64, 128}}. "
                f"Provided size '{size}' for {self.data_type} '{self.name}'."
            )


class AngleVariable(QasmVariable):
    """
    Fixed point angles
    """

    @property
    def data_type(self):
        return "angle"

    @property
    def value(self):
        if self._value is not None:
            return (2 * np.pi * self._value) / (2 ** self._size)

    def validate_value(self, value, size):
        try:
            self._value = int((value % (2 * np.pi)) / (2 * np.pi) * 2 ** self._size)
        except (ValueError, TypeError):
            raise ValueError(
                f"Not a valid value for {self.data_type} '{self.name}': {repr(value)}"
            )
