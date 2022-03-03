from typing import Any, List, Optional

from openqasm3.ast import ArrayLiteral, ClassicalType, IndexElement

from braket.default_simulator.openqasm.quantum_simulator import QuantumSimulator


class Table:
    def __init__(self, title):
        self._title = title
        self._dict = {}

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def items(self):
        return self._dict.items()

    def _longest_key_length(self):
        items = self.items()
        return max(len(key) for key, value in items) if items else None

    def __repr__(self):
        rows = [self._title]
        longest_key_length = self._longest_key_length()
        for item, value in self.items():
            rows.append(f"{item:<{longest_key_length}}\t{value}")
        return "\n".join(rows)


class ScopedTable(Table):
    def __init__(self, title):
        super().__init__(title)
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

    def __setitem__(self, key, value):
        self.current_scope[key] = value

    def items(self):
        items = {}
        for scope in reversed(self._scopes):
            for key, value in scope.items():
                if key not in items:
                    items[key] = value
        return items.items()

    def __repr__(self):
        rows = [self._title]
        longest_key_length = self._longest_key_length()
        for level, scope in enumerate(self._scopes):
            rows.append(f"SCOPE LEVEL {level}")
            for item, value in scope.items():
                rows.append(f"{item:<{longest_key_length}}\t{value}")
        return "\n".join(rows)


class SymbolTable(ScopedTable):
    class Symbol:
        def __init__(self, symbol_type: ClassicalType, const: bool = False):
            self.type = symbol_type
            self.const = const

        def __repr__(self):
            return f"Symbol<{self.type}, const={self.const}>"

    def __init__(self):
        super().__init__("Symbols")

    def add_symbol(
        self,
        name: str,
        symbol_type: ClassicalType,
        const: bool = False,
    ):
        self[name] = SymbolTable.Symbol(symbol_type, const)

    def get_symbol(self, name: str):
        return self[name]

    def get_type(self, name: str):
        return self.get_symbol(name).type

    def get_const(self, name: str):
        return self.get_symbol(name).const


class VariableTable(ScopedTable):
    def __init__(self):
        super().__init__("Data")

    def add_variable(self, name: str, value: Any):
        self[name] = value

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
        self.qubit_mapping = Table("Qubits")

    def __repr__(self):
        return f"{self.symbol_table}\n\n{self.variable_table}\n\n{self.qubit_mapping}"

    @property
    def num_qubits(self):
        return self.quantum_simulator.num_qubits

    def declare_variable(
        self,
        name: str,
        symbol_type: ClassicalType,
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

    def add_qubits(self, name: str, num_qubits: int):
        self.qubit_mapping[name] = range(self.num_qubits, self.num_qubits + num_qubits)
        self.quantum_simulator.add_qubits(num_qubits)

    def get_qubit_length(self, name: str):
        return len(self.qubit_mapping[name])

    def reset_qubits(self, qubits: str, index=None):
        print(qubits, index)
        target = self.qubit_mapping[qubits]
        if index is not None:
            target = target[index]
        self.quantum_simulator.reset_qubits(target)
