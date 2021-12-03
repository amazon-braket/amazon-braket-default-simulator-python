import numpy as np


class QasmVariable:

    def __init__(self, name, value=None, size=None):
        self.name = name
        self._value = value
        self._size = size
        if value is not None:
            self.validate_value(value, size)
        if size is not None:
            self.validate_size(size)

    @property
    def data_type(self):
        """ Name of the variable type """
        return ""

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
            except (AssertionError, ValueError):
                raise ValueError(
                    f"Invalid string '{value}' to initialize bit register '{self.name}'. "
                    "Provided string must be a binary string of length equal to "
                    f"given size '{size}'."
                )
        else:
            self._value = bool(value)