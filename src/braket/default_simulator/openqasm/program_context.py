from typing import Any

from openqasm.ast import ClassicalType


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
        return max(len(key) for key, value in self.items())

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
        def __init__(self, symbol_type: ClassicalType, const: bool = False):
            self.type = symbol_type
            self.const = const

        def __repr__(self):
            return f"Symbol<{self.type}, const={self.const}>"

    def add_symbol(self, name: str, symbol_type: ClassicalType, const: bool = False):
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


class ProgramContext:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.variable_table = VariableTable()

    def __repr__(self):
        return "Symbols\n" f"{self.symbol_table}\n\n" "Data\n" f"{self.variable_table}\n"
