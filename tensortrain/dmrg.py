"""Two-site DMRG algorithm."""
from __future__ import annotations

import logging

from typing import List, Tuple

import numpy as np
import tensornetwork as tn
import scipy.sparse.linalg as sla

import tensortrain.basics as tt

LOGGER = logging.getLogger(__name__)


def show(nodes: List[tn.Node]):
    """Draw tensor graph."""
    tn.to_graphviz(nodes).view()
    input("wait")


class DMRG(tt.Sweeper):
    """DMRG method to obtain ground-state."""

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
            self.update_ham_left(site+1, state_node=left)
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
            self.update_ham_right(site-1, state_node=right)
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
