from typing import Any, List, Optional, Sequence, Union

from openqasm3.ast import ArrayLiteral, ClassicalType, IndexElement

from braket.default_simulator.openqasm.quantum import QuantumSimulator, QubitType


class ScopedTable:
    def __init__(self):
        self._scopes = [{}]

    def push_scope(self):
        self._scopes.append({})

    def pop_scope(self):
        self._scopes.pop()

    @property
    def current_scope(self):
        return self._scopes[-1]

    def __getitem__(self, item):
        for scope in reversed(self._scopes):
            if item in scope:
                return scope[item]

    def items(self):
        items = {}
        for scope in reversed(self._scopes):
            for key, value in scope.items():
                if key not in items:
                    items[key] = value
        return items.items()

    def _longest_key_length(self):
        items = self.items()
        return max(len(key) for key, value in items) if items else None

    def __repr__(self):
        rows = []
        longest_key_length = self._longest_key_length()
        for level, scope in enumerate(self._scopes):
            rows.append(f"SCOPE LEVEL {level}")
            for item, value in scope.items():
                rows.append(f"{item:<{longest_key_length}}\t{value}")
        return "\n".join(rows)


class SymbolTable(ScopedTable):
    class Symbol:
        def __init__(self, symbol_type: Union[ClassicalType, QubitType], const: bool = False):
            self.type = symbol_type
            self.const = const

        def __repr__(self):
            return f"Symbol<{self.type}, const={self.const}>"

    def add_symbol(
        self,
        name: str,
        symbol_type: Union[ClassicalType, QubitType],
        const: bool = False,
    ):
        self.current_scope[name] = SymbolTable.Symbol(symbol_type, const)

    def get_symbol(self, name: str):
        return self[name]

    def get_type(self, name: str):
        return self.get_symbol(name).type

    def get_const(self, name: str):
        return self.get_symbol(name).const


class VariableTable(ScopedTable):
    def add_variable(self, name: str, value: Any):
        self.current_scope[name] = value

    def get_value(self, name: str):
        return self[name]

    def update_value(self, name: str, value: Any, indices: Optional[List[IndexElement]] = None):
        variable = self[name]
        if indices and not (isinstance(variable, ArrayLiteral)):
            raise TypeError(f"Cannot get index for variable of type {type(variable).__name__}")
        for i in indices:
            variable = variable.values[i]
        # if isinstance()


class ProgramContext:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.variable_table = VariableTable()
        self.quantum_simulator = QuantumSimulator()

    def __repr__(self):
        return f"Symbols\n{self.symbol_table}\n\nData\n{self.variable_table}\n"

    def declare_variable(
        self,
        name: str,
        symbol_type: Union[ClassicalType, QubitType],
        value: Optional[Any] = None,
        const: bool = False,
    ):
        self.symbol_table.add_symbol(name, symbol_type, const)
        self.variable_table.add_variable(name, value)

    def get_type(self, name: str):
        return self.symbol_table.get_type(name)

    def get_const(self, name: str):
        return self.symbol_table.get_const(name)

    def get_value(self, name: str):
        return self.variable_table.get_value(name)

    def update_value(self, name: str, value: Any, indices: Optional[List[IndexElement]] = None):
        return self.variable_table.update_value(name, value, indices)

    def add_qubits(self, num_qubits: int):
        self.quantum_simulator.add_qubits(num_qubits)

    def reset_qubits(self, target: Union[int, Sequence]):
        self.quantum_simulator.reset_qubits(target)
