"""Two-site DMRG algorithm."""
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


class DMRG:
    """DMRG method to obtain ground-state."""

    def __init__(self, mps: MPS, ham: MPO):
        """Use starting state and Hamiltonian."""
        if len(mps) != len(ham):
            raise ValueError("MPS and Hamiltonian size don't match.")
        self.mps = mps.copy()
        if mps.center is None:  # no orthogonalization
            mps.center = len(mps) - 1
        if mps.center != 0:  # right orthogonalize MPS
            mps.set_center(0)
        self.ham = ham.copy()
        self.ham_left: List[tn.Node] = [None] * len(ham)
        self.ham_left[0] = ham.left.copy()
        self.ham_right = self.build_right_ham()

    def build_right_ham(self) -> List[tn.Node]:
        """Create products for right Hamiltonian."""
        nsite = len(self.mps)
        mps = self.mps.copy()
        con = self.mps.copy(conjugate=True)
        ham = self.ham.copy()
        mpo = ham.nodes
        mpo_r = ham.right
        # connect the network
        for site in range(nsite):  # connect vertically
            tn.connect(mps[site]["phys"], mpo[site]["phys_in"])
            tn.connect(mpo[site]["phys_out"], con[site]["phys"])
        tn.connect(mps[-1]["right"], mpo_r["phys_in"])
        tn.connect(con[-1]["right"], mpo_r["phys_out"])

        # contract the network
        ham_right: List[tn.Node] = [None] * nsite
        ham_right[-1] = mpo_r.copy()
        for site in range(nsite-1, 0, -1):
            # show([mps[ii], mpo[ii], con[ii], mpo_r])
            mpo_r = tn.contractors.auto(
                [mps[site], mpo[site], con[site], mpo_r],
                output_edge_order=[nodes[site]["left"] for nodes in (mpo, mps, con)],
            )
            mpo_r.name = f"HR{site}"
            ham_right[site-1] = mpo_r.copy()
            ham_right[site-1].axis_names = ham_right[-1].axis_names
        return ham_right

    def sweep_2site_right(self, max_bond_dim: int, trunc_weight: float
                          ) -> Tuple[List[float], List[float]]:
        """Sweep from left to right, optimizing always two sites at once."""
        assert self.mps.center == 0, "To sweep right we start from the left"
        assert None not in self.ham_right[1:], "We need all right parts"
        energies: List[float] = []
        tws: List[float] = []
        for site in range(0, len(self.mps)-1):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site].copy(), self.ham[site].copy(),
                   self.ham[site+1].copy(), self.ham_right[site+1].copy()]
            chain(mpo)
            # TODO: set other arguments (we don't need that high accuracy):
            # ncv : Lanczos vectors
            # maxiter
            # tol : stopping criterion for accuracy

            # show(mpo)
            node1, node2 = self.mps[site:site+2]
            v0 = tn.contract_between(
                node1, node2,
                output_edge_order=[node1["left"], node1["phys"], node2["phys"], node2["right"]]
            )
            gs_energy, gs_vec = sla.eigsh(
                mpo_operator(mpo), k=1, which='SA', v0=v0.tensor.reshape(-1),
                # tol=1e-6
            )
            energies.append(gs_energy.item())
            dbl_node = tn.Node(gs_vec.reshape(v0.shape))
            # split the tensor and compress it moving center to the right
            left, rs, rvh, trunc_s = tn.split_node_full_svd(
                dbl_node, dbl_node[:2], dbl_node[2:],
                max_singular_values=max_bond_dim, max_truncation_err=trunc_weight,
                left_name=f"{site}L",
            )
            tws.append(np.sqrt(np.sum(trunc_s**2)))
            if tws[-1] > 0:
                rs.tensor /= np.sum(rs.tensor**2)
            left.add_axis_names(MS_AXES)
            right = tn.contract_between(
                rs, rvh, name=str(site+1), output_edge_order=[rs[0], *rvh[1:]],
                axis_names=MS_AXES,
            )
            self.mps.set_range(site, site+2, [left, right])
            self.mps.center += 1
            # create new left Hamiltonian
            mpol: List[tn.Node] = [self.ham_left[site].copy(), self.mps[site].copy(),
                                   self.mps[site].copy(conjugate=True), self.ham[site].copy()]
            tn.connect(mpol[1]["left"], mpol[0]["phys_in"])
            tn.connect(mpol[3]["left"], mpol[0]["right"])
            tn.connect(mpol[2]["left"], mpol[0]["phys_out"])
            tn.connect(mpol[1]["phys"], mpol[3]["phys_in"])
            tn.connect(mpol[2]["phys"], mpol[3]["phys_out"])
            self.ham_left[site+1] = tn.contractors.auto(
                mpol, output_edge_order=[mpol[3]["right"], mpol[1]["right"], mpol[2]["right"]]
            )
            self.ham_left[site+1].name = f"LH{site+1}"
            self.ham_left[site+1].axis_names = self.ham_left[site].axis_names
            LOGGER.debug("Right sweep: energy %e, bond-dim %3s, trunc %.3e",
                         energies[-1], rs.tensor.shape[0], tws[-1])
            if tws[-1] > trunc_weight:
                LOGGER.warning("Max. bond dim %3d between sites %d--%d. Truncation error %.3e.",
                               max_bond_dim, site-1, site, tws[-1])
        return energies, tws

    def sweep_2site_left(self, max_bond_dim: int, trunc_weight: float
                         ) -> Tuple[List[float], List[float]]:
        """Sweep from right to left, optimizing always two sites at once."""
        assert self.mps.center == len(self.mps) - 1, "To sweep right we start from the left"
        assert None not in self.ham_left[:-1], "We need all left parts"
        energies: List[float] = []
        tws: List[float] = []
        for site in range(len(self.mps)-1, 0, -1):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site-1].copy(), self.ham[site-1].copy(),
                   self.ham[site].copy(), self.ham_right[site].copy()]
            chain(mpo)
            node1, node2 = self.mps[site-1:site+1]
            v0 = tn.contract_between(
                node1, node2,
                output_edge_order=[node1["left"], node1["phys"], node2["phys"], node2["right"]]
            )
            gs_energy, gs_vec = sla.eigsh(
                mpo_operator(mpo), k=1, which='SA', v0=v0.tensor.reshape(-1),
                # tol=1e-6
            )
            energies.append(gs_energy.item())
            dbl_node = tn.Node(gs_vec.reshape(v0.shape))
            # split the tensor and compress it moving center to the right
            lu, ls, right, trunc_s = tn.split_node_full_svd(
                dbl_node, dbl_node[:2], dbl_node[2:],
                max_singular_values=max_bond_dim, max_truncation_err=trunc_weight,
                right_name=f"R{site}",
            )
            tws.append(np.sqrt(np.sum(trunc_s**2)))
            if tws[-1] > 0:
                ls.tensor /= np.sum(ls.tensor**2)
            right.add_axis_names(MS_AXES)
            left = tn.contract_between(
                lu, ls, name=str(site-1), output_edge_order=[*lu[:-1], ls[1]],
                axis_names=MS_AXES,
            )
            self.mps.set_range(site-1, site+1, [left, right])
            self.mps.center -= 1
            # create new right Hamiltonian
            mpor: List[tn.Node] = [self.ham_right[site].copy(), self.mps[site].copy(),
                                   self.mps[site].copy(conjugate=True), self.ham[site].copy()]
            tn.connect(mpor[1]["right"], mpor[0]["phys_in"])
            tn.connect(mpor[3]["right"], mpor[0]["left"])
            tn.connect(mpor[2]["right"], mpor[0]["phys_out"])
            tn.connect(mpor[1]["phys"], mpor[3]["phys_in"])
            tn.connect(mpor[2]["phys"], mpor[3]["phys_out"])
            self.ham_right[site-1] = tn.contractors.auto(
                mpor, output_edge_order=[mpor[3]["left"], mpor[1]["left"], mpor[2]["left"]]
            )
            self.ham_right[site-1].name = f"HR{site+1}"
            self.ham_right[site-1].axis_names = self.ham_right[site].axis_names
            LOGGER.debug("Left sweep:  energy %e, bond-dim %3s, trunc %.3e",
                         energies[-1], ls.tensor.shape[0], tws[-1])
            if tws[-1] > trunc_weight:
                LOGGER.warning("Max. bond dim %3d between sites %d--%d. Truncation error %.3e.",
                               max_bond_dim, site-1, site, tws[-1])
        return energies, tws

    def sweep_2site(self, max_bond_dim: int, trunc_weight: float):
        """Full DMRG sweep optimizing 2 sites at once."""
        energy_right, tw_right = self.sweep_2site_right(max_bond_dim, trunc_weight)
        energy_left, tw_left = self.sweep_2site_left(max_bond_dim, trunc_weight)
        return energy_right + energy_left, tw_right + tw_left

    def eval_ham2(self) -> float:
        """Evaulate ⟨ψ|H²|ψ⟩."""
        # do iteratively to save memory
        bra = self.mps.copy()
        ham1 = self.ham.copy()
        ham2 = self.ham.copy()
        ket = self.mps.copy(conjugate=True)
        # connect network
        tn.connect(ham1.left["phys_out"], ham2.left["phys_in"])
        tn.connect(bra[0]["left"], ham1.left["phys_in"])
        tn.connect(ket[0]["left"], ham2.left["phys_out"])
        for bra_s, ham1_s, ham2_s, ket_s in zip(bra, ham1, ham2, ket):
            tn.connect(bra_s["phys"], ham1_s["phys_in"])
            tn.connect(ham1_s["phys_out"], ham2_s["phys_in"])
            tn.connect(ket_s["phys"], ham2_s["phys_out"])
        tn.connect(bra[-1]["right"], ham1.right["phys_in"])
        tn.connect(ket[-1]["right"], ham2.right["phys_out"])
        tn.connect(ham1.right["phys_out"], ham2.right["phys_in"])

        # contract the network
        mpo_r = tn.contract_between(ham1.right, ham2.right)
        for bra_s, ham1_s, ham2_s, ket_s in zip(bra[::-1], ham1[::-1],
                                                ham2[::-1], ket[::-1]):
            ham12 = tn.contract_between(ham1_s, ham2_s)
            mpo_r = tn.contract_between(bra_s, mpo_r)
            mpo_r = tn.contract_between(ham12, mpo_r)
            mpo_r = tn.contract_between(ket_s, mpo_r)
        mpo_l = tn.contract_between(ham1.left, ham2.left)
        ham_sqr = tn.contract_between(mpo_l, mpo_r)
        return ham_sqr.tensor.item()


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
    """Make `mpo` into a linear operator for Lanczos."""
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

    return sla.LinearOperator(shape=(size, size), matvec=matvec_)


def setup_logging(level):
    """Set logging level and handler."""
    try:  # use colored log if available
        import colorlog  # pylint: disable=import-outside-toplevel
    except ImportError:  # use standard logging
        logging.basicConfig()
    else:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s:%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
