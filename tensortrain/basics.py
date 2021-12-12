"""Implement basic elements and operations for tensor-trains."""
from __future__ import annotations

import logging

from typing import Iterable, List, Optional
from dataclasses import dataclass, InitVar
from collections.abc import Sequence

import numpy as np
import tensornetwork as tn
import scipy.sparse.linalg as sla

LOGGER = logging.getLogger(__name__)

AXES_S = ("left", "phys", "right")  #: names of edges of a matrix state
AXES_O = ("left", "right", "phys_in", "phys_out")  #: names of edges of a matrix operator


def show(nodes: List[tn.Node]):
    """Draw tensor graph."""
    tn.to_graphviz(nodes).view()
    input("wait")


def _norm(node: tn.Node) -> float:
    """Calculate norm of `node` using the back-end."""
    return node.backend.norm(node.tensor)


def _sum(node: tn.Node, **kwds):
    return node.backend.sum(node.tensor, **kwds)


def _sqrt(node: tn.Node):
    return node.backend.sqrt(node.tensor)


def chain(nodes: List[tn.Node]) -> None:
    """Connect a chain of nodes 'left' to 'right'."""
    for left, right in zip(nodes[:-1], nodes[1:]):
        tn.connect(left["right"], right["left"])


def inner(a: State, b: State) -> np.number:
    """Calculate inner [scalar] product of two states `a` and `b`."""
    if len(a) != len(b):
        raise ValueError(f"Sizes of operators don't match ⟨{len(a)}, {len(b)}⟩")

    bra = a.copy(conjugate=True)
    ket = b.copy()

    left = tn.Node(np.ones([1, 1]), axis_names=["right_b", "right_k"])
    tn.connect(left["right_b"], bra[0]["left"])
    tn.connect(left["right_k"], ket[0]["left"])
    for bi, ki in zip(bra, ket):
        tn.connect(bi["phys"], ki["phys"])
        left = tn.contractors.auto(
            [left, bi, ki],
            output_edge_order=[bi["right"], ki["right"]],
        )
    tn.connect(left[0], left[1])
    return tn.contract_trace_edges(left).tensor.item()


@dataclass
class State(Sequence):
    """Container for the tensor-train state."""

    nodes: List[tn.Node]
    center: Optional[int] = None
    canonicalize: InitVar[bool] = True

    def __post_init__(self, canonicalize):
        """Handle canonicalization."""
        if canonicalize:  # FIXME: currently we ignore previously set center.
            self.center = len(self) - 1
            self.set_center(new_pos=0)

    @classmethod
    def from_random(cls, phys_dims: List[int], bond_dims: List[int], seed=None):
        """Create random tensor-train state.

        Parameters
        ----------
        phys_dims : List[int]
            Physical dimension of the sites.
        bond_dims : List[int]
            Initial bond dimension between the sites
        seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the BitGenerator. See `numpy.random.default_rng`.

        """
        if len(phys_dims) != len(bond_dims) + 1:
            raise ValueError(f"Number of bonds ({len(bond_dims)}) must be one"
                             f" smaller then number of nodes ({len(phys_dims)}).")
        bond_dims = [1] + bond_dims + [1]  # add dimensions for edge states
        rng = np.random.default_rng(seed)
        nodes: List[tn.Node] = [
            tn.Node(rng.standard_normal([bond_l, phys, bond_r]),
                    axis_names=AXES_S, name=str(ii))
            for ii, (bond_l, phys, bond_r)
            in enumerate(zip(bond_dims[:-1], phys_dims, bond_dims[1:]))
        ]
        chain(nodes)

        return cls(nodes, canonicalize=True)

    def set_center(self, new_pos: int) -> None:
        """Set the center of orthogonality `self.center` to `new_pos` using QR."""
        assert self.center is not None, """I don't want to handle this at the moment."""
        if not 0 <= new_pos < len(self):
            raise ValueError
        if new_pos == self.center:
            return
        if new_pos < self.center:
            for pos in range(self.center, new_pos, -1):
                node_r = self[pos]
                left, self.nodes[pos] = tn.split_node_rq(
                    node_r, right_name=f"R{pos}",
                    left_edges=[node_r['left']],
                    right_edges=[node_r['phys'], node_r['right']],
                )
                self.nodes[pos].add_axis_names(AXES_S)
                node_l = self.nodes[pos-1]
                self.nodes[pos-1] = tn.contract_between(node_l, left, name=str(pos-1),
                                                        axis_names=AXES_S)
                # normalize network
                self.nodes[pos-1].tensor /= _norm(self.nodes[pos-1])
            self.center = pos - 1
            return
        if new_pos > self.center:
            for pos in range(self.center, new_pos):
                node_l = self.nodes[pos]
                self.nodes[pos], right = tn.split_node_qr(
                    node_l, left_name=f"{pos}L",
                    left_edges=[node_l['left'], node_l['phys']],
                    right_edges=[node_l['right']],
                )
                self.nodes[pos].add_axis_names(AXES_S)
                node_r = self.nodes[pos+1]
                self.nodes[pos+1] = tn.contract_between(right, node_r, name=str(pos+1),
                                                        axis_names=AXES_S)
                # normalize network
                self.nodes[pos+1].tensor /= _norm(self.nodes[pos+1])
            self.center = pos + 1
            return
        raise NotImplementedError("This shouldn't have happened.")

    def set_node(self, pos: int, value: tn.Node) -> None:
        """Replace node at `pos` with `value` and connect it."""
        self.nodes[pos] = value
        if pos > 0:  # connect left
            left = self[pos - 1]
            if not left['right'].is_dangling():
                left['right'].disconnect()
            tn.connect(left['right'], self[pos]['left'])
        if pos < len(self) - 1:  # connect right
            right = self[pos + 1]
            if not right["left"].is_dangling():
                right["left"].disconnect()
            tn.connect(self[pos]["right"], right["left"])

    def set_range(self, start: int, stop: int, value: Iterable[tn.Node]) -> None:
        """Replace and connect nodes."""
        self.nodes[start:stop] = value
        if start > 0:  # connect left
            left = self[start - 1]
            if not left['right'].is_dangling():
                left['right'].disconnect()
            tn.connect(left['right'], self[start]['left'])
        if stop < len(self):  # connect right
            right = self[stop]
            if not right["left"].is_dangling():
                right["left"].disconnect()
            tn.connect(self[stop-1]["right"], right["left"])

    def __str__(self) -> str:
        """Print the state assuming it is canonicalized."""
        names = [node.name for node in self.nodes]
        phys = ['│' + ' '*(len(name) - 1) for name in names]
        line1 = '  '.join(phys)
        line2 = '──'.join(names)
        return line1 + '\n' + line2

    def __getitem__(self, item):
        """Directly access nodes."""
        return self.nodes[item]

    def __len__(self) -> int:
        """Give number of nodes."""
        return len(self.nodes)

    def copy(self, conjugate=False) -> State:
        """Create (shallow) copy."""
        nodes = tn.replicate_nodes(self.nodes, conjugate=conjugate)
        if conjugate:
            for node in nodes:
                node.name = node.name[:-1] if node.name[-1] == '*' else node.name + '*'
        state = self.__class__(nodes, center=self.center, canonicalize=False)
        return state


@dataclass
class Operator(Sequence):
    """Container for the tensor-train operator."""

    nodes: List[tn.Node]
    left: tn.Node
    right: tn.Node

    def __getitem__(self, item):
        """Directly access nodes."""
        return self.nodes[item]

    def __len__(self) -> int:
        """Give number of nodes."""
        return len(self.nodes)

    def copy(self) -> Operator:
        """Create (shallow) copy."""
        left, *nodes, right = tn.replicate_nodes([self.left] + self.nodes + [self.right])
        operator = self.__class__(nodes, left=left, right=right)
        return operator

    def __matmul__(self, other: Operator):
        """Connect and contract 'phys_in' of `self` with 'phys_out' of `other`."""
        if not isinstance(other, Operator):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"Sizes of operators don't match {len(self)} @ {len(other)}")

        def combine(mo1: tn.Node, mo2: tn.Node) -> tn.Node:
            tn.connect(mo1["phys_in"], mo2["phys_out"])
            return tn.contract_between(
                mo1, mo2,
                name=mo1.name+mo2.name,
                output_edge_order=[  # FIXME: order different than usual
                    mo2['left'], mo1['left'], mo2['phys_in'], mo1['phys_out'],
                    mo2["right"], mo1["right"]],
                axis_names=["left2", "left1", "phys_in", "phys_out", "right2", "right1"],
            )

        op1 = self.copy()
        op2 = other.copy()
        tn.connect(op1.left["phys_in"], op2.left["phys_out"])
        left = tn.contract_between(
            op1.left, op2.left, name=op1.left.name+op1.left.name,
            output_edge_order=[op2.left['right'], op1.left['right'],
                               op2.left['phys_in'], op1.left['phys_out']],
            axis_names=["right2", "right1", "phys_in", "phys_out"],
        )
        tn.connect(op1.right["phys_in"], op2.right["phys_out"])
        right = tn.contract_between(
            op1.right, op2.right, name=op1.right.name+op1.right.name,
            output_edge_order=[op2.right['left'], op1.right['left'],
                               op2.right['phys_in'], op1.right['phys_out']],
            axis_names=["left2", "left1", "phys_in", "phys_out"],
        )
        nodes = [combine(n1, n2) for n1, n2 in zip(op1, op2)]
        return Operator(nodes, left=left, right=right)


def _matvec(operator, vec_shape, path, vec):
    """Matrix-vector product of `operator`."""
    node = tn.Node(vec.reshape(vec_shape))
    oper_ = tn.replicate_nodes(operator)
    out_edges = [onode["phys_out"] for onode in oper_]
    for ii, onode in enumerate(oper_):
        tn.connect(node[ii], onode['phys_in'])
    # res = tn.contractors.auto(oper_ + [node], output_edge_order=out_edges)
    res = tn.contractors.contract_path(path, oper_ + [node], output_edge_order=out_edges)
    return res.tensor.reshape(-1)


def herm_linear_operator(operator: List[tn.Node]) -> sla.LinearOperator:
    """Create Hermitian linear operator from `operator`."""
    vec_shape = tuple(onode['phys_in'].dimension for onode in operator)
    size = np.product(vec_shape)
    # Pre-compute optimal path
    testnode = tn.Node(np.empty(vec_shape))
    oper_ = tn.replicate_nodes(operator)
    for ii, onode in enumerate(oper_):
        tn.connect(testnode[ii], onode['phys_in'])
    path = tn.contractors.path_solver('optimal', nodes=oper_+[testnode])

    def matvec_(vec):
        return _matvec(operator, vec_shape=vec_shape, path=path, vec=vec)

    # Hamiltonian is Hermitian -> rmatvec=matvec
    return sla.LinearOperator(shape=(size, size), matvec=matvec_, rmatvec=matvec_)
