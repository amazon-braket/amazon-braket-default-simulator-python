"""
Copied over and debugged from the openqasm repo
"""

from dataclasses import asdict
from functools import singledispatchmethod
from typing import Generic, Optional, TypeVar

from openqasm.ast import QASMNode

T = TypeVar("T")


class QASMVisitor(Generic[T]):
    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.

    Modified from the implementation in ast.py in the Python standard library.
    We added the context argument to the visit method. It allows the visitor
    to hold temporary state while visiting the nodes.

    The optional context argument in visit/generic_visit methods can be used to hold temporary
    information that we do not want to hold in either the AST or the visitor themselves.
    """

    @singledispatchmethod
    def visit(self, node: QASMNode, context: Optional[T] = None):
        """Visit a node. Called if no explicit visitor function exists for a node."""
        for value in asdict(node).values():
            if not isinstance(value, list):
                value = [value]
            for item in value:
                if isinstance(item, QASMNode):
                    self.visit(item, context)


class QASMTransformer(QASMVisitor[T]):
    """
    A :class:`QASMVisitor` subclass that walks the abstract syntax tree and
    allows modification of nodes.

    Modified from the implementation in ast.py in the Python standard library
    """

    def visit(self, node: QASMNode, context: Optional[T] = None) -> QASMNode:
        if node is None:
            return None
        for field, old_value in list(node.__dict__.items()):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, QASMNode):
                        value = self.visit(value, context) if context else self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, QASMNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, QASMNode):
                new_node = self.visit(old_value, context) if context else self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node
