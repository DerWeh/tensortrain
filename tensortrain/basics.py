"""Implement basic elements and operations for tensor-trains."""
from __future__ import annotations

import logging

from collections.abc import Sequence
from dataclasses import dataclass, InitVar
from functools import partial
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensornetwork as tn
import scipy.sparse.linalg as sla

LOGGER = logging.getLogger(__name__)

AXES_S = ("left", "phys", "right")  #: names of edges of a state node
AXES_O = ("left", "right", "phys_in", "phys_out")  #: names of edges of a operator node
AXES_L = ("right", "phys_in", "phys_out")  #: names of edges of left operator node
AXES_R = ("left", "phys_in", "phys_out")  #: names of edges of right operator node


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
            self.set_center(new_pos=0, normalize=True)

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

    def set_center(self, new_pos: int, normalize=False) -> None:
        """Set the center of orthogonality `self.center` to `new_pos` using QR."""
        assert self.center is not None, "I don't want to handle this at the moment."
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
                if normalize:  # normalize in every set to avoid accumulation
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
                if normalize:  # normalize in every set to avoid accumulation
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

    def __eq__(self, o):
        """Equality check for easier debugging, requires all elements to be equal.

        States with different gauges are **not** considered as equal!

        """
        if not isinstance(o, State):
            return False
        if len(self) != len(o):
            return False
        for self_node, o_node in zip(self, o):
            if not np.all(self_node.tensor == o_node.tensor):
                return False
        return True

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


def herm_linear_operator_slow(operator: List[tn.Node]) -> sla.LinearOperator:
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


def _to_uname(axis: str, num: int):
    if axis == "phys_out":
        return -num  # external leg
    if axis == "phys_in":
        return num
    if axis == "right":
        return f"c{num}"
    if axis == "left":
        return f"c{num-1}"
    raise AttributeError


def _resolve_path(path: List[Tuple[int, int]], structure: List[tuple]):
    structure = structure.copy()
    resolved: list = []
    for nodes in path:
        n2, n1 = sorted(nodes)
        ax1 = set(structure.pop(n1))
        ax2 = set(structure.pop(n2))
        resolved.extend(ax1 & ax2)
        structure.append(ax1 ^ ax2)
    return resolved


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
    operator = [op.tensor for op in oper_]

    # assumes chain
    num = len(oper_)
    structure = []
    for ii, op in enumerate(oper_, start=1):
        structure.append(tuple(_to_uname(ax, ii) for ax in op.axis_names))
    structure.append(tuple(range(1, num+1)))
    ncon = partial(tn.ncon,
                   network_structure=structure,
                   con_order=_resolve_path(path, structure),
                   out_order=tuple(range(-1, -num-1, -1)),
                   check_network=False,
                   )

    def matvec_(vec):
        return ncon(operator + [vec.reshape(vec_shape)]).reshape(-1)

    # Hamiltonian is Hermitian -> rmatvec=matvec
    return sla.LinearOperator(shape=(size, size), matvec=matvec_, rmatvec=matvec_)


def trace(operator: List[tn.Node]) -> np.number:
    """Calculate the trace of the network `operator`."""
    operator = tn.replicate_nodes(operator)
    for node in operator:
        tn.connect(node["phys_in"], node["phys_out"])
    traced = [tn.contract_trace_edges(node) for node in operator]
    return tn.contractors.auto(traced).tensor.item()


class Sweeper:
    """Base class for sweeping algorithms, implementing effective Hamiltonians."""

    def __init__(self, state: State, ham: Operator):
        """Use starting state and Hamiltonian."""
        if len(state) != len(ham):
            raise ValueError("State and Hamiltonian size don't match.")
        self.state = state.copy()
        if state.center is None:  # no orthogonalization
            state.center = len(state) - 1
        if state.center != 0:  # right orthogonalize MPS
            state.set_center(0)
        self.ham = ham.copy()
        self.ham_left: List[tn.Node] = [None] * len(ham)
        self.ham_left[0] = ham.left.copy()
        self.ham_right = self.build_right_ham()

    def build_right_ham(self) -> List[tn.Node]:
        """Create products for right Hamiltonian."""
        nsite = len(self.state)
        ket = self.state.copy()
        bra = self.state.copy(conjugate=True)
        ham_full = self.ham.copy()
        ham = ham_full.nodes
        ham_r = ham_full.right
        # connect the network
        for site in range(nsite):  # connect vertically
            tn.connect(ket[site]["phys"], ham[site]["phys_in"])
            tn.connect(bra[site]["phys"], ham[site]["phys_out"])
        tn.connect(ket[-1]["right"], ham_r["phys_in"])
        tn.connect(bra[-1]["right"], ham_r["phys_out"])

        # contract the network
        ham_rights: List[tn.Node] = [None] * nsite
        ham_rights[-1] = ham_r.copy()
        for site in range(nsite-1, 0, -1):
            # show([mps[ii], ham[ii], con[ii], ham_r])
            ham_r = tn.contractors.auto(
                [ket[site], ham[site], bra[site], ham_r],
                output_edge_order=[nodes[site]["left"] for nodes in (ham, ket, bra)],
            )
            ham_r.name = f"HR{site}"
            ham_r.axis_names = AXES_R
            ham_rights[site-1] = ham_r.copy()
        return ham_rights

    def update_ham_left(self, site: int, state_node: tn.Node):
        """Calculate left Hamiltonian at `site` from ``site-1`` and `state_node`."""
        if site == 0:
            raise ValueError("Left Hamiltonian for site 0 is fixed.")
        ham_left = self.ham_left[site-1].copy()
        bra = state_node.copy(conjugate=True)
        ham = self.ham[site-1].copy()
        ket = state_node.copy()
        tn.connect(bra["left"], ham_left["phys_out"])
        tn.connect(ham["left"], ham_left["right"])
        tn.connect(ket["left"], ham_left["phys_in"])
        tn.connect(bra["phys"], ham["phys_out"])
        tn.connect(ket["phys"], ham["phys_in"])
        self.ham_left[site] = tn.contractors.auto(
            [ham_left, bra, ham, ket],
            output_edge_order=[ham["right"], ket["right"], bra["right"]]  # match AXES_L
        )
        self.ham_left[site].name = f"LH{site}"
        self.ham_left[site].axis_names = AXES_L

    def update_ham_right(self, site: int, state_node: tn.Node):
        """Calculate right Hamiltonian at `site` from ``site-1`` and `state_node`."""
        if site == len(self.state) - 1:
            raise ValueError("Right Hamiltonian for last site is fixed.")

        ham_right = self.ham_right[site+1].copy()
        bra = state_node.copy(conjugate=True)
        ham = self.ham[site+1].copy()
        ket = state_node.copy()

        # create new right Hamiltonian
        tn.connect(bra["right"], ham_right["phys_out"])
        tn.connect(ham["right"], ham_right["left"])
        tn.connect(ket["right"], ham_right["phys_in"])
        tn.connect(bra["phys"], ham["phys_out"])
        tn.connect(ket["phys"], ham["phys_in"])
        self.ham_right[site] = tn.contractors.auto(
            [ham_right, bra, ham, ket],
            output_edge_order=[ham["left"], ket["left"], bra["left"]]
        )
        self.ham_right[site].name = f"HR{site}"
        self.ham_right[site].axis_names = AXES_R
