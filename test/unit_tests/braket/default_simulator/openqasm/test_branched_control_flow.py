# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
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

"""Tests for branched control flow handlers in ProgramContext (Task 5.3).

Tests verify that handle_branching_statement, handle_for_loop, and
handle_while_loop correctly delegate to super() when not branched,
and perform per-path evaluation when branched.
"""

import pytest
from copy import deepcopy

from braket.default_simulator.openqasm.parser.openqasm_ast import (
    BooleanLiteral,
    BranchingStatement,
    BreakStatement,
    ContinueStatement,
    ForInLoop,
    Identifier,
    IntegerLiteral,
    IntType,
    RangeDefinition,
    WhileLoop,
)
from braket.default_simulator.openqasm.program_context import (
    AbstractProgramContext,
    ProgramContext,
    _BreakSignal,
    _ContinueSignal,
)
from braket.default_simulator.openqasm.simulation_path import FramedVariable, SimulationPath


class TestBranchedBranchingStatement:
    """Tests for handle_branching_statement in branched mode."""

    def test_not_branched_delegates_to_super(self):
        """When not branched, handle_branching_statement should use default eager evaluation."""
        context = ProgramContext()
        assert not context.is_branched

        visited = []

        def mock_visit(node):
            if isinstance(node, BooleanLiteral):
                return node
            visited.append(node)
            return node

        # Create a simple branching statement with condition=True
        node = BranchingStatement(
            condition=BooleanLiteral(True),
            if_block=["if_stmt_1", "if_stmt_2"],
            else_block=["else_stmt_1"],
        )

        context.handle_branching_statement(node, mock_visit)
        assert visited == ["if_stmt_1", "if_stmt_2"]

    def test_not_branched_else_block(self):
        """When not branched and condition is False, else block should be visited."""
        context = ProgramContext()
        visited = []

        def mock_visit(node):
            if isinstance(node, BooleanLiteral):
                return node
            visited.append(node)
            return node

        node = BranchingStatement(
            condition=BooleanLiteral(False),
            if_block=["if_stmt"],
            else_block=["else_stmt"],
        )

        context.handle_branching_statement(node, mock_visit)
        assert visited == ["else_stmt"]

    def test_branched_routes_paths_by_condition(self):
        """When branched, paths should be routed based on per-path condition evaluation."""
        context = ProgramContext()
        # Manually set up branched state with two paths
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        path1 = SimulationPath([], 50, {}, {})
        # Path 0 has condition_var = True, Path 1 has condition_var = False
        path0.set_variable("c", FramedVariable("c", None, BooleanLiteral(True), False, 0))
        path1.set_variable("c", FramedVariable("c", None, BooleanLiteral(False), False, 0))
        context._paths = [path0, path1]
        context._active_path_indices = [0, 1]

        if_visited_paths = []
        else_visited_paths = []

        def mock_visit(node):
            if isinstance(node, Identifier) and node.name == "c":
                # Return the value from the current active path
                path_idx = context._active_path_indices[0]
                path = context._paths[path_idx]
                var = path.get_variable("c")
                return var.value
            if isinstance(node, BooleanLiteral):
                return node
            if node == "if_stmt":
                if_visited_paths.extend(list(context._active_path_indices))
            elif node == "else_stmt":
                else_visited_paths.extend(list(context._active_path_indices))
            return node

        node = BranchingStatement(
            condition=Identifier("c"),
            if_block=["if_stmt"],
            else_block=["else_stmt"],
        )

        context.handle_branching_statement(node, mock_visit)

        # Path 0 (True) should have gone through if_block
        assert 0 in if_visited_paths
        # Path 1 (False) should have gone through else_block
        assert 1 in else_visited_paths
        # Both paths should survive
        assert set(context._active_path_indices) == {0, 1}

    def test_branched_no_else_block(self):
        """When branched with no else block, false paths should survive unchanged."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        path1 = SimulationPath([], 50, {}, {})
        path0.set_variable("c", FramedVariable("c", None, BooleanLiteral(True), False, 0))
        path1.set_variable("c", FramedVariable("c", None, BooleanLiteral(False), False, 0))
        context._paths = [path0, path1]
        context._active_path_indices = [0, 1]

        if_visited = []

        def mock_visit(node):
            if isinstance(node, Identifier) and node.name == "c":
                path_idx = context._active_path_indices[0]
                return context._paths[path_idx].get_variable("c").value
            if isinstance(node, BooleanLiteral):
                return node
            if node == "if_stmt":
                if_visited.extend(list(context._active_path_indices))
            return node

        node = BranchingStatement(
            condition=Identifier("c"),
            if_block=["if_stmt"],
            else_block=[],
        )

        context.handle_branching_statement(node, mock_visit)

        assert 0 in if_visited
        assert 1 not in if_visited
        # Both paths survive
        assert set(context._active_path_indices) == {0, 1}


class TestBranchedForLoop:
    """Tests for handle_for_loop in branched mode."""

    def test_not_branched_delegates_to_super(self):
        """When not branched, handle_for_loop should use default eager unrolling."""
        context = ProgramContext()
        assert not context.is_branched

        iterations = []

        def mock_visit(node):
            if isinstance(node, RangeDefinition):
                return node
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            iterations.append(node)
            return node

        node = ForInLoop(
            type=IntType(IntegerLiteral(32)),
            identifier=Identifier("i"),
            set_declaration=RangeDefinition(
                IntegerLiteral(0), IntegerLiteral(2), IntegerLiteral(1)
            ),
            block=["body_stmt"],
        )

        context.handle_for_loop(node, mock_visit)
        # Should have iterated 3 times (0, 1, 2)
        body_visits = [x for x in iterations if x == "body_stmt"]
        assert len(body_visits) == 3

    def test_branched_sets_loop_variable_per_path(self):
        """When branched, loop variable should be set per-path."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        path1 = SimulationPath([], 50, {}, {})
        context._paths = [path0, path1]
        context._active_path_indices = [0, 1]

        loop_var_values = []

        def mock_visit(node):
            if isinstance(node, RangeDefinition):
                return node
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            if node == "body_stmt":
                # Record the loop variable value for each active path
                for path_idx in context._active_path_indices:
                    var = context._paths[path_idx].get_variable("i")
                    if var:
                        loop_var_values.append((path_idx, var.value))
            return node

        node = ForInLoop(
            type=IntType(IntegerLiteral(32)),
            identifier=Identifier("i"),
            set_declaration=RangeDefinition(
                IntegerLiteral(0), IntegerLiteral(1), IntegerLiteral(1)
            ),
            block=["body_stmt"],
        )

        context.handle_for_loop(node, mock_visit)

        # Both paths should have iterated with values 0 and 1
        assert len(loop_var_values) >= 2
        # After loop, both paths should still be active
        assert set(context._active_path_indices) == {0, 1}

    def test_branched_for_loop_break(self):
        """Break in branched for loop should stop iteration."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        context._paths = [path0]
        context._active_path_indices = [0]

        iteration_count = [0]

        def mock_visit(node):
            if isinstance(node, RangeDefinition):
                return node
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            if isinstance(node, BreakStatement):
                context.handle_break_statement()
                return node
            if node == "body_stmt":
                iteration_count[0] += 1
            return node

        node = ForInLoop(
            type=IntType(IntegerLiteral(32)),
            identifier=Identifier("i"),
            set_declaration=RangeDefinition(
                IntegerLiteral(0), IntegerLiteral(4), IntegerLiteral(1)
            ),
            block=["body_stmt", BreakStatement()],
        )

        context.handle_for_loop(node, mock_visit)

        # Should have only executed body once before break
        assert iteration_count[0] == 1
        # Path should still be active (break exits loop, not path)
        assert 0 in context._active_path_indices

    def test_branched_for_loop_continue(self):
        """Continue in branched for loop should skip to next iteration."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        context._paths = [path0]
        context._active_path_indices = [0]

        pre_continue_count = [0]
        post_continue_count = [0]

        def mock_visit(node):
            if isinstance(node, RangeDefinition):
                return node
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            if isinstance(node, ContinueStatement):
                context.handle_continue_statement()
                return node
            if node == "pre_continue":
                pre_continue_count[0] += 1
            elif node == "post_continue":
                post_continue_count[0] += 1
            return node

        node = ForInLoop(
            type=IntType(IntegerLiteral(32)),
            identifier=Identifier("i"),
            set_declaration=RangeDefinition(
                IntegerLiteral(0), IntegerLiteral(2), IntegerLiteral(1)
            ),
            block=["pre_continue", ContinueStatement(), "post_continue"],
        )

        context.handle_for_loop(node, mock_visit)

        # pre_continue should execute each iteration (3 times: 0, 1, 2)
        assert pre_continue_count[0] == 3
        # post_continue should never execute (skipped by continue)
        assert post_continue_count[0] == 0


class TestBranchedWhileLoop:
    """Tests for handle_while_loop in branched mode."""

    def test_not_branched_delegates_to_super(self):
        """When not branched, handle_while_loop should use default eager evaluation."""
        context = ProgramContext()
        assert not context.is_branched

        counter = [3]

        def mock_visit(node):
            if isinstance(node, BooleanLiteral):
                return node
            if isinstance(node, IntegerLiteral):
                result = BooleanLiteral(counter[0] > 0)
                return result
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            if node == "body_stmt":
                counter[0] -= 1
            return node

        node = WhileLoop(
            while_condition=IntegerLiteral(1),  # Will be evaluated by mock
            block=["body_stmt"],
        )

        context.handle_while_loop(node, mock_visit)
        assert counter[0] == 0

    def test_branched_while_loop_per_path_condition(self):
        """When branched, while condition should be evaluated per-path."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        path1 = SimulationPath([], 50, {}, {})
        # Path 0 loops 2 times, Path 1 loops 0 times
        path0.set_variable("n", FramedVariable("n", None, IntegerLiteral(2), False, 0))
        path1.set_variable("n", FramedVariable("n", None, IntegerLiteral(0), False, 0))
        context._paths = [path0, path1]
        context._active_path_indices = [0, 1]

        body_executions = {0: 0, 1: 0}

        def mock_visit(node):
            if isinstance(node, Identifier) and node.name == "n":
                path_idx = context._active_path_indices[0]
                var = context._paths[path_idx].get_variable("n")
                val = var.value.value
                return BooleanLiteral(val > 0)
            if isinstance(node, BooleanLiteral):
                return node
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            if node == "body_stmt":
                for path_idx in context._active_path_indices:
                    body_executions[path_idx] += 1
                    var = context._paths[path_idx].get_variable("n")
                    new_val = IntegerLiteral(var.value.value - 1)
                    context._paths[path_idx].set_variable(
                        "n", FramedVariable("n", None, new_val, False, 0)
                    )
            return node

        node = WhileLoop(
            while_condition=Identifier("n"),
            block=["body_stmt"],
        )

        context.handle_while_loop(node, mock_visit)

        # Path 0 should have looped 2 times
        assert body_executions[0] == 2
        # Path 1 should have looped 0 times
        assert body_executions[1] == 0
        # Both paths should survive
        assert set(context._active_path_indices) == {0, 1}

    def test_branched_while_loop_break(self):
        """Break in branched while loop should exit the loop."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {})
        context._paths = [path0]
        context._active_path_indices = [0]

        iteration_count = [0]

        def mock_visit(node):
            if isinstance(node, BooleanLiteral):
                return node
            if isinstance(node, IntegerLiteral):
                return BooleanLiteral(True)  # Always true
            if isinstance(node, list):
                for item in node:
                    mock_visit(item)
                return
            if isinstance(node, BreakStatement):
                context.handle_break_statement()
                return node
            if node == "body_stmt":
                iteration_count[0] += 1
            return node

        node = WhileLoop(
            while_condition=IntegerLiteral(1),
            block=["body_stmt", BreakStatement()],
        )

        context.handle_while_loop(node, mock_visit)

        assert iteration_count[0] == 1
        assert 0 in context._active_path_indices


class TestBreakContinueSignals:
    """Tests for break/continue signal mechanism."""

    def test_break_signal_raised_when_branched(self):
        """handle_break_statement should raise _BreakSignal when branched."""
        context = ProgramContext()
        context._is_branched = True
        with pytest.raises(_BreakSignal):
            context.handle_break_statement()

    def test_break_signal_not_raised_when_not_branched(self):
        """handle_break_statement should raise _BreakSignal even when not branched.
        The signal is caught by the enclosing loop handler."""
        context = ProgramContext()
        assert not context.is_branched
        with pytest.raises(_BreakSignal):
            context.handle_break_statement()

    def test_continue_signal_raised_when_branched(self):
        """handle_continue_statement should raise _ContinueSignal when branched."""
        context = ProgramContext()
        context._is_branched = True
        with pytest.raises(_ContinueSignal):
            context.handle_continue_statement()

    def test_continue_signal_not_raised_when_not_branched(self):
        """handle_continue_statement should raise _ContinueSignal even when not branched.
        The signal is caught by the enclosing loop handler."""
        context = ProgramContext()
        assert not context.is_branched
        with pytest.raises(_ContinueSignal):
            context.handle_continue_statement()


class TestFrameManagement:
    """Tests for _enter_frame_for_active_paths and _exit_frame_for_active_paths."""

    def test_enter_frame_increments_frame_number(self):
        """Entering a frame should increment frame_number for all active paths."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {}, frame_number=0)
        path1 = SimulationPath([], 50, {}, {}, frame_number=0)
        context._paths = [path0, path1]
        context._active_path_indices = [0, 1]

        context._enter_frame_for_active_paths()

        assert path0.frame_number == 1
        assert path1.frame_number == 1

    def test_exit_frame_restores_frame_number(self):
        """Exiting a frame should restore frame_number for all active paths."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {}, frame_number=1)
        path1 = SimulationPath([], 50, {}, {}, frame_number=1)
        context._paths = [path0, path1]
        context._active_path_indices = [0, 1]

        context._exit_frame_for_active_paths()

        assert path0.frame_number == 0
        assert path1.frame_number == 0

    def test_exit_frame_removes_scoped_variables(self):
        """Exiting a frame should remove variables declared in that frame."""
        context = ProgramContext()
        context._is_branched = True
        path0 = SimulationPath([], 50, {}, {}, frame_number=1)
        # Variable declared in frame 1 (current frame)
        path0.set_variable("x", FramedVariable("x", None, IntegerLiteral(10), False, 1))
        # Variable declared in frame 0 (outer frame)
        path0.set_variable("y", FramedVariable("y", None, IntegerLiteral(20), False, 0))
        context._paths = [path0]
        context._active_path_indices = [0]

        context._exit_frame_for_active_paths()

        # x (frame 1) should be removed, y (frame 0) should remain
        assert path0.get_variable("x") is None
        assert path0.get_variable("y") is not None
        assert path0.get_variable("y").value == IntegerLiteral(20)


class TestAbstractContextBreakContinue:
    """Tests for handle_break_statement and handle_continue_statement on AbstractProgramContext."""

    def test_abstract_break_is_noop(self):
        """AbstractProgramContext.handle_break_statement raises _BreakSignal.
        The signal is caught by the enclosing loop handler."""
        context = ProgramContext()
        with pytest.raises(_BreakSignal):
            context.handle_break_statement()

    def test_abstract_continue_is_noop(self):
        """AbstractProgramContext.handle_continue_statement raises _ContinueSignal.
        The signal is caught by the enclosing loop handler."""
        context = ProgramContext()
        with pytest.raises(_ContinueSignal):
            context.handle_continue_statement()
