"""Two-site DMRG algorithm."""
from __future__ import annotations

import logging

from typing import List, Tuple

import numpy as np
import tensornetwork as tn
import scipy.sparse.linalg as sla

import tensortrain as tt

LOGGER = logging.getLogger(__name__)


def show(nodes: List[tn.Node]):
    """Draw tensor graph."""
    tn.to_graphviz(nodes).view()
    input("wait")


class DMRG:
    """DMRG method to obtain ground-state."""

    def __init__(self, state: tt.State, ham: tt.Operator):
        """Use starting state and Hamiltonian."""
        if len(state) != len(ham):
            raise ValueError("MPS and Hamiltonian size don't match.")
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
        ham = self.ham.copy()
        mpo = ham.nodes
        mpo_r = ham.right
        # connect the network
        for site in range(nsite):  # connect vertically
            tn.connect(ket[site]["phys"], mpo[site]["phys_in"])
            tn.connect(mpo[site]["phys_out"], bra[site]["phys"])
        tn.connect(ket[-1]["right"], mpo_r["phys_in"])
        tn.connect(bra[-1]["right"], mpo_r["phys_out"])

        # contract the network
        ham_right: List[tn.Node] = [None] * nsite
        ham_right[-1] = mpo_r.copy()
        for site in range(nsite-1, 0, -1):
            # show([mps[ii], mpo[ii], con[ii], mpo_r])
            mpo_r = tn.contractors.auto(
                [ket[site], mpo[site], bra[site], mpo_r],
                output_edge_order=[nodes[site]["left"] for nodes in (mpo, ket, bra)],
            )
            mpo_r.name = f"HR{site}"
            ham_right[site-1] = mpo_r.copy()
            ham_right[site-1].axis_names = ham_right[-1].axis_names
        return ham_right

    def sweep_2site_right(self, max_bond_dim: int, trunc_weight: float
                          ) -> Tuple[List[float], List[float]]:
        """Sweep from left to right, optimizing always two sites at once."""
        assert self.state.center == 0, "To sweep right we start from the left"
        assert None not in self.ham_right[1:], "We need all right parts"
        energies: List[float] = []
        tws: List[float] = []
        for site in range(0, len(self.state)-1):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site].copy(), self.ham[site].copy(),
                   self.ham[site+1].copy(), self.ham_right[site+1].copy()]
            tt.chain(mpo)
            # TODO: set other arguments (we don't need that high accuracy):
            # ncv : Lanczos vectors
            # maxiter
            # tol : stopping criterion for accuracy

            # show(mpo)
            node1, node2 = self.state[site:site+2]
            v0 = tn.contract_between(
                node1, node2,
                output_edge_order=[node1["left"], node1["phys"], node2["phys"], node2["right"]]
            )
            gs_energy, gs_vec = sla.eigsh(
                tt.herm_linear_operator(mpo), k=1, which='SA', v0=v0.tensor.reshape(-1),
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
            left.add_axis_names(tt.AXES_S)
            right = tn.contract_between(
                rs, rvh, name=str(site+1), output_edge_order=[rs[0], *rvh[1:]],
                axis_names=tt.AXES_S,
            )
            self.state.set_range(site, site+2, [left, right])
            self.state.center += 1
            # create new left Hamiltonian
            mpol: List[tn.Node] = [self.ham_left[site].copy(), self.state[site].copy(),
                                   self.state[site].copy(conjugate=True), self.ham[site].copy()]
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
        assert self.state.center == len(self.state) - 1, "To sweep right we start from the left"
        assert None not in self.ham_left[:-1], "We need all left parts"
        energies: List[float] = []
        tws: List[float] = []
        for site in range(len(self.state)-1, 0, -1):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site-1].copy(), self.ham[site-1].copy(),
                   self.ham[site].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            node1, node2 = self.state[site-1:site+1]
            v0 = tn.contract_between(
                node1, node2,
                output_edge_order=[node1["left"], node1["phys"], node2["phys"], node2["right"]]
            )
            gs_energy, gs_vec = sla.eigsh(
                tt.herm_linear_operator(mpo), k=1, which='SA', v0=v0.tensor.reshape(-1),
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
            right.add_axis_names(tt.AXES_S)
            left = tn.contract_between(
                lu, ls, name=str(site-1), output_edge_order=[*lu[:-1], ls[1]],
                axis_names=tt.AXES_S,
            )
            self.state.set_range(site-1, site+1, [left, right])
            self.state.center -= 1
            # create new right Hamiltonian
            mpor: List[tn.Node] = [self.ham_right[site].copy(), self.state[site].copy(),
                                   self.state[site].copy(conjugate=True), self.ham[site].copy()]
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
        bra = self.state.copy()
        ham1 = self.ham.copy()
        ham2 = self.ham.copy()
        ket = self.state.copy(conjugate=True)
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
