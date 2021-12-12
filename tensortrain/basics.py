"""Implement basic elements and operations for tensor-trains."""
from __future__ import annotations

import logging

from typing import Iterable, List, Optional, Tuple
from dataclasses import dataclass, InitVar
from collections.abc import Sequence

import numpy as np
import tensornetwork as tn
import scipy.sparse.linalg as sla

LOGGER = logging.getLogger(__name__)

MS_AXES = ("left", "phys", "right")  #: names of edges of a matrix state
MO_AXES = ("left", "right", "phys_in", "phys_out")  #: names of edges of a matrix operator


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


def inner(a: MPS, b: MPS) -> np.number:
    """Calculate inner (scalar) product of two states `a` and `b`."""
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
class MPS(Sequence):
    """Container for the matrix product state."""

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
        """Create random MPS.

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
                    axis_names=MS_AXES, name=str(ii))
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
                self.nodes[pos].add_axis_names(MS_AXES)
                node_l = self.nodes[pos-1]
                self.nodes[pos-1] = tn.contract_between(node_l, left, name=str(pos-1),
                                                        axis_names=MS_AXES)
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
                self.nodes[pos].add_axis_names(MS_AXES)
                node_r = self.nodes[pos+1]
                self.nodes[pos+1] = tn.contract_between(right, node_r, name=str(pos+1),
                                                        axis_names=MS_AXES)
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

    def copy(self, conjugate=False) -> MPS:
        """Create (shallow) copy."""
        nodes = tn.replicate_nodes(self.nodes, conjugate=conjugate)
        if conjugate:
            for node in nodes:
                node.name = node.name[:-1] if node.name[-1] == '*' else node.name + '*'
        mps = self.__class__(nodes, center=self.center, canonicalize=False)
        return mps


@dataclass
class MPO(Sequence):
    """Container for the matrix product operator."""

    nodes: List[tn.Node]
    left: tn.Node
    right: tn.Node

    def __getitem__(self, item):
        """Directly access nodes."""
        return self.nodes[item]

    def __len__(self) -> int:
        """Give number of nodes."""
        return len(self.nodes)

    def copy(self) -> MPO:
        """Create (shallow) copy."""
        left, *nodes, right = tn.replicate_nodes([self.left] + self.nodes + [self.right])
        mpo = self.__class__(nodes, left=left, right=right)
        return mpo

    def __matmul__(self, other: MPO):
        """Connect and contract 'phys_in' of `self` with 'phys_out' of `other`."""
        if not isinstance(other, MPO):
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

        mpo1 = self.copy()
        mpo2 = other.copy()
        tn.connect(mpo1.left["phys_in"], mpo2.left["phys_out"])
        left = tn.contract_between(
            mpo1.left, mpo2.left, name=mpo1.left.name+mpo1.left.name,
            output_edge_order=[mpo2.left['right'], mpo1.left['right'],
                               mpo2.left['phys_in'], mpo1.left['phys_out']],
            axis_names=["right2", "right1", "phys_in", "phys_out"],
        )
        tn.connect(mpo1.right["phys_in"], mpo2.right["phys_out"])
        right = tn.contract_between(
            mpo1.right, mpo2.right, name=mpo1.right.name+mpo1.right.name,
            output_edge_order=[mpo2.right['left'], mpo1.right['left'],
                               mpo2.right['phys_in'], mpo1.right['phys_out']],
            axis_names=["left2", "left1", "phys_in", "phys_out"],
        )
        nodes = [combine(n1, n2) for n1, n2 in zip(mpo1, mpo2)]
        return MPO(nodes, left=left, right=right)


def _matvec(mpo, vec_shape, path, vec):
    """Matrix-vector product of `mpo` for Lanczos."""
    node = tn.Node(vec.reshape(vec_shape))
    mpo_ = tn.replicate_nodes(mpo)
    out_edges = [onode["phys_out"] for onode in mpo_]
    for ii, onode in enumerate(mpo_):
        tn.connect(node[ii], onode['phys_in'])
    # res = tn.contractors.auto(mpo_ + [node], output_edge_order=out_edges)
    res = tn.contractors.contract_path(path, mpo_ + [node], output_edge_order=out_edges)
    return res.tensor.reshape(-1)


def mpo_operator(mpo: List[tn.Node]) -> sla.LinearOperator:
    """Make `mpo` into a Hermitian linear operator for Lanczos."""
    vec_shape = tuple(onode['phys_in'].dimension for onode in mpo)
    size = np.product(vec_shape)
    # Pre-compute optimal path
    testnode = tn.Node(np.empty(vec_shape))
    mpo_ = tn.replicate_nodes(mpo)
    for ii, onode in enumerate(mpo_):
        tn.connect(testnode[ii], onode['phys_in'])
    path = tn.contractors.path_solver('optimal', nodes=mpo_+[testnode])

    def matvec_(vec):
        return _matvec(mpo, vec_shape=vec_shape, path=path, vec=vec)

    # Hamiltonian is Hermitian -> rmatvec=matvec
    return sla.LinearOperator(shape=(size, size), matvec=matvec_, rmatvec=matvec_)
