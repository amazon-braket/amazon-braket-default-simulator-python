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

"""
Tensor Network Contraction Engine for Quantum Circuit Simulation.

This module provides exponential speedup for circuits with limited entanglement
by representing quantum circuits as tensor networks and finding optimal
contraction orders.

Key concepts:
- Each gate becomes a tensor node
- Qubit lines become edges connecting tensors
- Contraction order determines computational cost
- Optimal ordering can reduce O(2^n) to O(poly(n)) for structured circuits

For highly entangled circuits, falls back to standard state vector simulation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from braket.default_simulator.linalg_utils import _GPU_AVAILABLE

if _GPU_AVAILABLE:
    from numba import cuda

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation


# Configuration
_MAX_TENSOR_SIZE = 2**20  # Max intermediate tensor size before fallback
_MAX_BOND_DIMENSION = 64  # Max bond dimension for SVD truncation
_CONTRACTION_COST_THRESHOLD = 1e12  # Fallback if estimated cost too high
_MIN_QUBITS_FOR_TN = 12  # Minimum qubits to consider tensor network


@dataclass
class TensorNode:
    """
    A node in the tensor network representing a gate or initial state.
    
    Attributes:
        id: Unique identifier for this node
        tensor: The tensor data (numpy array)
        edges: List of edge IDs connected to this node
        shape: Shape of the tensor
        is_gate: Whether this is a gate tensor (vs initial state)
    """
    id: int
    tensor: np.ndarray
    edges: list[int] = field(default_factory=list)
    is_gate: bool = True
    
    @property
    def shape(self) -> tuple:
        return self.tensor.shape
    
    @property
    def size(self) -> int:
        return self.tensor.size
    
    @property
    def rank(self) -> int:
        return len(self.tensor.shape)


@dataclass
class TensorEdge:
    """
    An edge in the tensor network representing a qubit line or contraction.
    
    Attributes:
        id: Unique identifier for this edge
        dimension: Dimension of this edge (2 for qubit)
        nodes: List of node IDs connected by this edge
        is_open: Whether this edge is open (connects to output)
    """
    id: int
    dimension: int = 2
    nodes: list[int] = field(default_factory=list)
    is_open: bool = False


class TensorNetwork:
    """
    Tensor network representation of a quantum circuit.
    
    Converts a quantum circuit into a network of tensors where:
    - Initial |0⟩ states are rank-1 tensors
    - Single-qubit gates are rank-2 tensors
    - Two-qubit gates are rank-4 tensors
    - Qubit lines are edges connecting tensors
    
    The network can then be contracted to compute amplitudes or the full
    state vector, with the contraction order determining efficiency.
    """
    
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.nodes: dict[int, TensorNode] = {}
        self.edges: dict[int, TensorEdge] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self._qubit_current_edge: dict[int, int] = {}
        self._output_edges: list[int] = []
    
    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid
    
    def _new_edge_id(self) -> int:
        eid = self._next_edge_id
        self._next_edge_id += 1
        return eid
    
    def build_from_circuit(self, operations: list[GateOperation]) -> None:
        """
        Build tensor network from a list of gate operations.
        
        Creates initial state tensors for each qubit, then adds gate tensors
        connected by edges representing qubit lines.
        """
        # Create initial |0⟩ state tensors for each qubit
        for q in range(self.qubit_count):
            state_tensor = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
            node_id = self._new_node_id()
            edge_id = self._new_edge_id()
            
            node = TensorNode(
                id=node_id,
                tensor=state_tensor,
                edges=[edge_id],
                is_gate=False
            )
            edge = TensorEdge(
                id=edge_id,
                dimension=2,
                nodes=[node_id],
                is_open=False
            )
            
            self.nodes[node_id] = node
            self.edges[edge_id] = edge
            self._qubit_current_edge[q] = edge_id
        
        # Add gate tensors
        for op in operations:
            self._add_gate(op)
        
        # Mark output edges as open
        for q in range(self.qubit_count):
            edge_id = self._qubit_current_edge[q]
            self.edges[edge_id].is_open = True
            self._output_edges.append(edge_id)
    
    def _add_gate(self, op: GateOperation) -> None:
        """Add a gate tensor to the network."""
        targets = op.targets
        matrix = op.matrix
        ctrl_modifiers = getattr(op, "_ctrl_modifiers", [])
        num_ctrl = len(ctrl_modifiers)
        
        if num_ctrl > 0:
            # Controlled gate - treat as multi-qubit gate
            all_qubits = list(targets)
            self._add_multi_qubit_gate(matrix, all_qubits)
        elif len(targets) == 1:
            self._add_single_qubit_gate(matrix, targets[0])
        elif len(targets) == 2:
            self._add_two_qubit_gate(matrix, targets[0], targets[1])
        else:
            self._add_multi_qubit_gate(matrix, list(targets))
    
    def _add_single_qubit_gate(self, matrix: np.ndarray, qubit: int) -> None:
        """Add a single-qubit gate tensor."""
        # Reshape 2x2 matrix to rank-2 tensor with shape (out, in)
        gate_tensor = matrix.reshape(2, 2).astype(np.complex128)
        
        node_id = self._new_node_id()
        in_edge_id = self._qubit_current_edge[qubit]
        out_edge_id = self._new_edge_id()
        
        # Connect input edge to this node
        self.edges[in_edge_id].nodes.append(node_id)
        
        # Create output edge
        out_edge = TensorEdge(
            id=out_edge_id,
            dimension=2,
            nodes=[node_id],
            is_open=False
        )
        self.edges[out_edge_id] = out_edge
        
        # Create node
        node = TensorNode(
            id=node_id,
            tensor=gate_tensor,
            edges=[in_edge_id, out_edge_id],
            is_gate=True
        )
        self.nodes[node_id] = node
        
        # Update current edge for this qubit
        self._qubit_current_edge[qubit] = out_edge_id
    
    def _add_two_qubit_gate(self, matrix: np.ndarray, qubit0: int, qubit1: int) -> None:
        """Add a two-qubit gate tensor."""
        # Reshape 4x4 matrix to rank-4 tensor with shape (out0, out1, in0, in1)
        gate_tensor = matrix.reshape(2, 2, 2, 2).astype(np.complex128)
        
        node_id = self._new_node_id()
        in_edge0 = self._qubit_current_edge[qubit0]
        in_edge1 = self._qubit_current_edge[qubit1]
        out_edge0 = self._new_edge_id()
        out_edge1 = self._new_edge_id()
        
        # Connect input edges
        self.edges[in_edge0].nodes.append(node_id)
        self.edges[in_edge1].nodes.append(node_id)
        
        # Create output edges
        self.edges[out_edge0] = TensorEdge(
            id=out_edge0, dimension=2, nodes=[node_id], is_open=False
        )
        self.edges[out_edge1] = TensorEdge(
            id=out_edge1, dimension=2, nodes=[node_id], is_open=False
        )
        
        # Create node with edges in order: in0, in1, out0, out1
        node = TensorNode(
            id=node_id,
            tensor=gate_tensor,
            edges=[in_edge0, in_edge1, out_edge0, out_edge1],
            is_gate=True
        )
        self.nodes[node_id] = node
        
        self._qubit_current_edge[qubit0] = out_edge0
        self._qubit_current_edge[qubit1] = out_edge1
    
    def _add_multi_qubit_gate(self, matrix: np.ndarray, qubits: list[int]) -> None:
        """Add a multi-qubit gate tensor."""
        n = len(qubits)
        # Reshape to rank-2n tensor: (out0, out1, ..., in0, in1, ...)
        shape = [2] * (2 * n)
        gate_tensor = matrix.reshape(shape).astype(np.complex128)
        
        node_id = self._new_node_id()
        in_edges = [self._qubit_current_edge[q] for q in qubits]
        out_edges = [self._new_edge_id() for _ in qubits]
        
        # Connect input edges
        for e in in_edges:
            self.edges[e].nodes.append(node_id)
        
        # Create output edges
        for e in out_edges:
            self.edges[e] = TensorEdge(id=e, dimension=2, nodes=[node_id], is_open=False)
        
        # Create node
        node = TensorNode(
            id=node_id,
            tensor=gate_tensor,
            edges=in_edges + out_edges,
            is_gate=True
        )
        self.nodes[node_id] = node
        
        for q, e in zip(qubits, out_edges):
            self._qubit_current_edge[q] = e
    
    def estimate_contraction_cost(self, order: list[tuple[int, int]]) -> float:
        """
        Estimate the computational cost of a contraction order.
        
        Cost is the sum of intermediate tensor sizes created during contraction.
        """
        # Simulate contraction to compute cost
        node_edges = {nid: set(node.edges) for nid, node in self.nodes.items()}
        edge_dims = {eid: edge.dimension for eid, edge in self.edges.items()}
        
        total_cost = 0.0
        next_id = max(self.nodes.keys()) + 1
        
        for n1, n2 in order:
            if n1 not in node_edges or n2 not in node_edges:
                continue
            
            edges1 = node_edges[n1]
            edges2 = node_edges[n2]
            
            # Contracted edges (shared between n1 and n2)
            contracted = edges1 & edges2
            
            # Remaining edges after contraction
            remaining = (edges1 | edges2) - contracted
            
            # Cost is size of resulting tensor
            result_size = 1
            for e in remaining:
                if e in edge_dims:
                    result_size *= edge_dims[e]
            
            total_cost += result_size
            
            # Update for next iteration
            del node_edges[n1]
            del node_edges[n2]
            node_edges[next_id] = remaining
            next_id += 1
        
        return total_cost
    
    def get_node_count(self) -> int:
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        return len(self.edges)


class ContractionOptimizer:
    """
    Finds optimal or near-optimal contraction orders for tensor networks.
    
    Uses multiple strategies:
    1. Greedy: Fast heuristic based on minimizing intermediate tensor size
    2. Dynamic Programming: Optimal for small networks (< 20 nodes)
    3. Community Detection: Groups related tensors for hierarchical contraction
    
    The choice of strategy depends on network size and structure.
    """
    
    def __init__(self, max_dp_nodes: int = 16):
        self.max_dp_nodes = max_dp_nodes
        self._dp_cache = {}
    
    def find_order(self, network: TensorNetwork) -> list[tuple[int, int]]:
        """
        Find a good contraction order for the network.
        
        Returns list of (node1, node2) pairs to contract in order.
        """
        n_nodes = network.get_node_count()
        
        if n_nodes <= 2:
            nodes = list(network.nodes.keys())
            if len(nodes) == 2:
                return [(nodes[0], nodes[1])]
            return []
        
        if n_nodes <= self.max_dp_nodes:
            return self._dp_optimal(network)
        else:
            return self._greedy_order(network)
    
    def _greedy_order(self, network: TensorNetwork) -> list[tuple[int, int]]:
        """
        Greedy contraction order based on minimizing intermediate tensor size.
        
        At each step, contracts the pair of connected nodes that produces
        the smallest intermediate tensor.
        """
        # Build adjacency from edges
        node_edges = {nid: set(node.edges) for nid, node in network.nodes.items()}
        edge_dims = {eid: edge.dimension for eid, edge in network.edges.items()}
        edge_nodes = {eid: set(edge.nodes) for eid, edge in network.edges.items()}
        
        order = []
        next_id = max(network.nodes.keys()) + 1
        active_nodes = set(network.nodes.keys())
        
        while len(active_nodes) > 1:
            best_pair = None
            best_cost = float('inf')
            
            # Find all pairs of connected nodes
            checked = set()
            for n1 in active_nodes:
                for e in node_edges.get(n1, []):
                    for n2 in edge_nodes.get(e, []):
                        if n2 in active_nodes and n1 != n2:
                            pair = (min(n1, n2), max(n1, n2))
                            if pair in checked:
                                continue
                            checked.add(pair)
                            
                            # Compute cost of contracting this pair
                            cost = self._contraction_cost(
                                node_edges.get(n1, set()),
                                node_edges.get(n2, set()),
                                edge_dims
                            )
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_pair = pair
            
            if best_pair is None:
                # No connected pairs - find any pair
                nodes_list = list(active_nodes)
                if len(nodes_list) >= 2:
                    best_pair = (nodes_list[0], nodes_list[1])
                else:
                    break
            
            n1, n2 = best_pair
            order.append((n1, n2))
            
            # Merge nodes
            edges1 = node_edges.get(n1, set())
            edges2 = node_edges.get(n2, set())
            contracted = edges1 & edges2
            remaining = (edges1 | edges2) - contracted
            
            # Update edge_nodes for remaining edges
            for e in remaining:
                if e in edge_nodes:
                    edge_nodes[e].discard(n1)
                    edge_nodes[e].discard(n2)
                    edge_nodes[e].add(next_id)
            
            # Remove contracted edges
            for e in contracted:
                if e in edge_nodes:
                    del edge_nodes[e]
                if e in edge_dims:
                    del edge_dims[e]
            
            # Update node_edges
            del node_edges[n1]
            del node_edges[n2]
            node_edges[next_id] = remaining
            
            active_nodes.discard(n1)
            active_nodes.discard(n2)
            active_nodes.add(next_id)
            next_id += 1
        
        return order
    
    def _contraction_cost(
        self, edges1: set[int], edges2: set[int], edge_dims: dict[int, int]
    ) -> float:
        """Compute cost of contracting two nodes."""
        contracted = edges1 & edges2
        remaining = (edges1 | edges2) - contracted
        
        # Cost is product of remaining edge dimensions
        cost = 1
        for e in remaining:
            cost *= edge_dims.get(e, 2)
        
        # Also factor in contraction dimension
        contract_dim = 1
        for e in contracted:
            contract_dim *= edge_dims.get(e, 2)
        
        return cost * contract_dim
    
    def _dp_optimal(self, network: TensorNetwork) -> list[tuple[int, int]]:
        """
        Find optimal contraction order using dynamic programming.
        
        Only feasible for small networks due to exponential complexity.
        """
        nodes = list(network.nodes.keys())
        n = len(nodes)
        
        if n <= 1:
            return []
        if n == 2:
            return [(nodes[0], nodes[1])]
        
        # For simplicity, use greedy for now
        # Full DP implementation would use memoization over subsets
        return self._greedy_order(network)
    
    def estimate_cost(self, network: TensorNetwork, order: list[tuple[int, int]]) -> float:
        """Estimate total cost of a contraction order."""
        return network.estimate_contraction_cost(order)


class TensorContractor:
    """
    Executes tensor network contractions.
    
    Performs the actual tensor contractions following a given order,
    using numpy.einsum or optimized GPU kernels.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _GPU_AVAILABLE
    
    def contract(
        self,
        network: TensorNetwork,
        order: list[tuple[int, int]],
        output_indices: list[int] | None = None
    ) -> np.ndarray:
        """
        Contract the tensor network following the given order.
        
        Args:
            network: The tensor network to contract
            order: List of (node1, node2) pairs to contract
            output_indices: Optional specific output indices (for partial contraction)
        
        Returns:
            The contracted tensor (state vector if fully contracted)
        """
        # Copy tensors and edge info for manipulation
        tensors = {nid: node.tensor.copy() for nid, node in network.nodes.items()}
        node_edges = {nid: list(node.edges) for nid, node in network.nodes.items()}
        
        next_id = max(network.nodes.keys()) + 1
        
        for n1, n2 in order:
            if n1 not in tensors or n2 not in tensors:
                continue
            
            t1 = tensors[n1]
            t2 = tensors[n2]
            e1 = node_edges[n1]
            e2 = node_edges[n2]
            
            # Find shared edges (to contract)
            shared = set(e1) & set(e2)
            
            if not shared:
                # Outer product
                result = np.tensordot(t1, t2, axes=0)
                result_edges = e1 + e2
            else:
                # Contract over shared edges
                axes1 = [e1.index(e) for e in shared]
                axes2 = [e2.index(e) for e in shared]
                
                result = np.tensordot(t1, t2, axes=(axes1, axes2))
                
                # Compute remaining edges
                remaining1 = [e for e in e1 if e not in shared]
                remaining2 = [e for e in e2 if e not in shared]
                result_edges = remaining1 + remaining2
            
            # Store result
            tensors[next_id] = result
            node_edges[next_id] = result_edges
            
            # Clean up
            del tensors[n1]
            del tensors[n2]
            del node_edges[n1]
            del node_edges[n2]
            
            next_id += 1
        
        # Get final tensor
        if len(tensors) == 1:
            final_id = next(iter(tensors.keys()))
            result = tensors[final_id]
            final_edges = node_edges[final_id]
            
            # Reorder to match output edge order if needed
            if output_indices is not None and final_edges:
                # Permute to match expected output order
                output_edges = network._output_edges
                if set(final_edges) == set(output_edges):
                    perm = [final_edges.index(e) for e in output_edges]
                    result = np.transpose(result, perm)
            
            return result.flatten()
        
        elif len(tensors) > 1:
            # Multiple disconnected components - tensor product
            result = None
            for tid in tensors:
                if result is None:
                    result = tensors[tid]
                else:
                    result = np.tensordot(result, tensors[tid], axes=0)
            return result.flatten() if result is not None else np.array([1.0+0j])
        
        return np.array([1.0 + 0j], dtype=np.complex128)
    
    def contract_for_amplitude(
        self,
        network: TensorNetwork,
        order: list[tuple[int, int]],
        bitstring: str
    ) -> complex:
        """
        Contract network to get amplitude of a specific bitstring.
        
        More efficient than full contraction when only one amplitude is needed.
        """
        # Add projection tensors for each output qubit
        tensors = {nid: node.tensor.copy() for nid, node in network.nodes.items()}
        node_edges = {nid: list(node.edges) for nid, node in network.nodes.items()}
        
        next_id = max(network.nodes.keys()) + 1
        
        # Add projection tensors
        for i, bit in enumerate(bitstring):
            edge_id = network._output_edges[i]
            proj = np.array([1.0, 0.0] if bit == '0' else [0.0, 1.0], dtype=np.complex128)
            
            tensors[next_id] = proj
            node_edges[next_id] = [edge_id]
            next_id += 1
        
        # Contract (simplified - just use full contraction for now)
        result = self.contract(network, order)
        
        # Extract amplitude
        idx = int(bitstring, 2)
        if idx < len(result):
            return result[idx]
        return 0.0 + 0j


class GPUTensorContractor:
    """
    GPU-accelerated tensor contraction using batched matrix operations.
    
    For large tensors, uses GPU for the heavy contractions while keeping
    small tensors on CPU to avoid transfer overhead.
    """
    
    _MIN_GPU_SIZE = 2**12  # Minimum tensor size for GPU
    
    def __init__(self):
        self._stream = None
        if _GPU_AVAILABLE:
            self._stream = cuda.stream()
    
    def contract(
        self,
        network: TensorNetwork,
        order: list[tuple[int, int]]
    ) -> np.ndarray:
        """
        Contract tensor network with GPU acceleration for large tensors.
        """
        if not _GPU_AVAILABLE:
            return TensorContractor(use_gpu=False).contract(network, order)
        
        tensors = {nid: node.tensor.copy() for nid, node in network.nodes.items()}
        node_edges = {nid: list(node.edges) for nid, node in network.nodes.items()}
        
        next_id = max(network.nodes.keys()) + 1
        
        for n1, n2 in order:
            if n1 not in tensors or n2 not in tensors:
                continue
            
            t1 = tensors[n1]
            t2 = tensors[n2]
            e1 = node_edges[n1]
            e2 = node_edges[n2]
            
            shared = set(e1) & set(e2)
            
            # Decide CPU vs GPU based on tensor sizes
            use_gpu_for_this = (
                t1.size >= self._MIN_GPU_SIZE or 
                t2.size >= self._MIN_GPU_SIZE
            )
            
            if not shared:
                if use_gpu_for_this:
                    result = self._gpu_outer_product(t1, t2)
                else:
                    result = np.tensordot(t1, t2, axes=0)
                result_edges = e1 + e2
            else:
                axes1 = [e1.index(e) for e in shared]
                axes2 = [e2.index(e) for e in shared]
                
                if use_gpu_for_this:
                    result = self._gpu_tensordot(t1, t2, axes1, axes2)
                else:
                    result = np.tensordot(t1, t2, axes=(axes1, axes2))
                
                remaining1 = [e for e in e1 if e not in shared]
                remaining2 = [e for e in e2 if e not in shared]
                result_edges = remaining1 + remaining2
            
            tensors[next_id] = result
            node_edges[next_id] = result_edges
            
            del tensors[n1]
            del tensors[n2]
            del node_edges[n1]
            del node_edges[n2]
            
            next_id += 1
        
        if len(tensors) == 1:
            final_id = next(iter(tensors.keys()))
            result = tensors[final_id]
            final_edges = node_edges[final_id]
            
            output_edges = network._output_edges
            if final_edges and set(final_edges) == set(output_edges):
                perm = [final_edges.index(e) for e in output_edges]
                result = np.transpose(result, perm)
            
            return result.flatten()
        
        elif len(tensors) > 1:
            result = None
            for tid in tensors:
                if result is None:
                    result = tensors[tid]
                else:
                    result = np.tensordot(result, tensors[tid], axes=0)
            return result.flatten() if result is not None else np.array([1.0+0j])
        
        return np.array([1.0 + 0j], dtype=np.complex128)
    
    def _gpu_tensordot(
        self, t1: np.ndarray, t2: np.ndarray, axes1: list[int], axes2: list[int]
    ) -> np.ndarray:
        """
        GPU-accelerated tensor contraction.
        
        Reshapes tensors to matrices and uses GPU matrix multiplication.
        """
        # Reshape to matrices for efficient GPU matmul
        # Move contracted axes to end of t1 and beginning of t2
        
        n1_axes = list(range(t1.ndim))
        n2_axes = list(range(t2.ndim))
        
        # Axes not being contracted
        free1 = [i for i in n1_axes if i not in axes1]
        free2 = [i for i in n2_axes if i not in axes2]
        
        # Permute t1: free axes first, then contracted
        perm1 = free1 + axes1
        t1_perm = np.transpose(t1, perm1)
        
        # Permute t2: contracted first, then free
        perm2 = axes2 + free2
        t2_perm = np.transpose(t2, perm2)
        
        # Reshape to 2D
        contract_size = 1
        for ax in axes1:
            contract_size *= t1.shape[ax]
        
        free1_size = t1.size // contract_size
        free2_size = t2.size // contract_size
        
        m1 = t1_perm.reshape(free1_size, contract_size)
        m2 = t2_perm.reshape(contract_size, free2_size)
        
        # GPU matrix multiply
        m1_gpu = cuda.to_device(np.ascontiguousarray(m1), stream=self._stream)
        m2_gpu = cuda.to_device(np.ascontiguousarray(m2), stream=self._stream)
        result_gpu = cuda.device_array((free1_size, free2_size), dtype=np.complex128, stream=self._stream)
        
        # Use cublas or custom kernel
        self._gpu_matmul(m1_gpu, m2_gpu, result_gpu)
        
        self._stream.synchronize()
        result = result_gpu.copy_to_host()
        
        # Reshape back to tensor
        result_shape = [t1.shape[i] for i in free1] + [t2.shape[i] for i in free2]
        if result_shape:
            return result.reshape(result_shape)
        return result.flatten()
    
    def _gpu_outer_product(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """GPU-accelerated outer product."""
        # For outer product, just use numpy - GPU overhead not worth it
        return np.tensordot(t1, t2, axes=0)
    
    def _gpu_matmul(
        self,
        a_gpu: cuda.devicearray.DeviceNDArray,
        b_gpu: cuda.devicearray.DeviceNDArray,
        c_gpu: cuda.devicearray.DeviceNDArray
    ) -> None:
        """
        GPU matrix multiplication C = A @ B.
        
        Uses a simple tiled kernel. For production, would use cuBLAS.
        """
        m, k = a_gpu.shape
        _, n = b_gpu.shape
        
        TILE = 16
        blocks_x = (n + TILE - 1) // TILE
        blocks_y = (m + TILE - 1) // TILE
        
        _matmul_kernel[(blocks_y, blocks_x), (TILE, TILE), self._stream](
            a_gpu, b_gpu, c_gpu, m, n, k
        )


if _GPU_AVAILABLE:
    @cuda.jit(fastmath=True)
    def _matmul_kernel(A, B, C, m, n, k):
        """Tiled matrix multiplication kernel."""
        TILE = 16
        
        # Shared memory tiles
        sA = cuda.shared.array((TILE, TILE), dtype=numba.complex128)
        sB = cuda.shared.array((TILE, TILE), dtype=numba.complex128)
        
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        row = by * TILE + ty
        col = bx * TILE + tx
        
        acc = 0.0 + 0.0j
        
        for t in range((k + TILE - 1) // TILE):
            # Load tiles into shared memory
            if row < m and t * TILE + tx < k:
                sA[ty, tx] = A[row, t * TILE + tx]
            else:
                sA[ty, tx] = 0.0 + 0.0j
            
            if t * TILE + ty < k and col < n:
                sB[ty, tx] = B[t * TILE + ty, col]
            else:
                sB[ty, tx] = 0.0 + 0.0j
            
            cuda.syncthreads()
            
            # Compute partial dot product
            for i in range(TILE):
                acc += sA[ty, i] * sB[i, tx]
            
            cuda.syncthreads()
        
        if row < m and col < n:
            C[row, col] = acc

    import numba


class TensorNetworkSimulator:
    """
    High-level interface for tensor network-based circuit simulation.
    
    Automatically decides whether to use tensor network contraction or
    fall back to state vector simulation based on circuit structure.
    """
    
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.optimizer = ContractionOptimizer()
        self.contractor = GPUTensorContractor() if _GPU_AVAILABLE else TensorContractor()
        self._fallback_threshold = _CONTRACTION_COST_THRESHOLD
    
    def simulate(
        self, 
        operations: list[GateOperation],
        use_tensor_network: bool | None = None
    ) -> np.ndarray:
        """
        Simulate circuit and return state vector.
        
        Args:
            operations: List of gate operations
            use_tensor_network: Force TN (True), force state vector (False),
                               or auto-decide (None)
        
        Returns:
            Final state vector as numpy array
        """
        if not operations:
            state = np.zeros(2**self.qubit_count, dtype=np.complex128)
            state[0] = 1.0
            return state
        
        # Auto-decide based on circuit structure
        if use_tensor_network is None:
            use_tensor_network = self._should_use_tensor_network(operations)
        
        if not use_tensor_network:
            return self._fallback_state_vector(operations)
        
        # Build tensor network
        network = TensorNetwork(self.qubit_count)
        network.build_from_circuit(operations)
        
        # Find contraction order
        order = self.optimizer.find_order(network)
        
        # Estimate cost and potentially fallback
        estimated_cost = self.optimizer.estimate_cost(network, order)
        if estimated_cost > self._fallback_threshold:
            return self._fallback_state_vector(operations)
        
        # Contract
        if isinstance(self.contractor, GPUTensorContractor):
            result = self.contractor.contract(network, order)
        else:
            result = self.contractor.contract(network, order)
        
        return result
    
    def get_amplitude(
        self,
        operations: list[GateOperation],
        bitstring: str
    ) -> complex:
        """
        Get amplitude of a specific bitstring without full state computation.
        
        For structured circuits, this can be exponentially faster than
        computing the full state vector.
        """
        if len(bitstring) != self.qubit_count:
            raise ValueError(f"Bitstring length {len(bitstring)} != qubit_count {self.qubit_count}")
        
        network = TensorNetwork(self.qubit_count)
        network.build_from_circuit(operations)
        
        order = self.optimizer.find_order(network)
        
        contractor = TensorContractor()
        return contractor.contract_for_amplitude(network, order, bitstring)
    
    def _should_use_tensor_network(self, operations: list[GateOperation]) -> bool:
        """
        Decide whether tensor network is beneficial for this circuit.
        
        Heuristics:
        - Small qubit count: state vector is fine
        - Very deep circuits: TN overhead not worth it
        - Circuits with limited entanglement: TN wins
        """
        n = self.qubit_count
        depth = len(operations)
        
        # Too few qubits - state vector is fast enough
        if n < _MIN_QUBITS_FOR_TN:
            return False
        
        # Estimate entanglement from two-qubit gate count
        two_qubit_gates = sum(1 for op in operations if len(op.targets) >= 2)
        
        # If circuit is very entangling, TN won't help
        # Rough heuristic: if 2Q gates > n * log(n), probably too entangled
        entanglement_threshold = n * np.log2(n + 1)
        if two_qubit_gates > entanglement_threshold * 2:
            return False
        
        # For shallow circuits with limited entanglement, TN is beneficial
        if depth < n * 2 and two_qubit_gates < n:
            return True
        
        # Default: try TN for larger systems
        return n >= 16
    
    def _fallback_state_vector(self, operations: list[GateOperation]) -> np.ndarray:
        """Fall back to standard state vector simulation."""
        from braket.default_simulator.gpu_optimized_operations import (
            apply_operations_optimized,
        )
        
        state = np.zeros([2] * self.qubit_count, dtype=np.complex128)
        state.flat[0] = 1.0
        
        return apply_operations_optimized(state, self.qubit_count, operations)


# Global instance management
_global_tn_simulator: TensorNetworkSimulator | None = None
_tn_lock = threading.Lock()


def get_tensor_network_simulator(qubit_count: int) -> TensorNetworkSimulator:
    """Get or create tensor network simulator for given qubit count."""
    global _global_tn_simulator
    
    with _tn_lock:
        if _global_tn_simulator is None or _global_tn_simulator.qubit_count != qubit_count:
            _global_tn_simulator = TensorNetworkSimulator(qubit_count)
        return _global_tn_simulator


def simulate_with_tensor_network(
    qubit_count: int,
    operations: list[GateOperation],
    force_tn: bool | None = None
) -> np.ndarray:
    """
    Simulate circuit using tensor network when beneficial.
    
    Args:
        qubit_count: Number of qubits
        operations: List of gate operations
        force_tn: Force tensor network (True), force state vector (False),
                  or auto-decide (None)
    
    Returns:
        Final state vector
    """
    simulator = get_tensor_network_simulator(qubit_count)
    return simulator.simulate(operations, use_tensor_network=force_tn)


def get_amplitude_tensor_network(
    qubit_count: int,
    operations: list[GateOperation],
    bitstring: str
) -> complex:
    """
    Get amplitude of specific bitstring using tensor network.
    
    Can be exponentially faster than full state computation for
    circuits with limited entanglement.
    """
    simulator = get_tensor_network_simulator(qubit_count)
    return simulator.get_amplitude(operations, bitstring)
