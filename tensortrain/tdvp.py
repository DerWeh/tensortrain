"""Time-depended variational principle for tensor trains."""
import logging

from typing import List

import numpy as np
import tensornetwork as tn

import tensortrain.basics as tt

from tensortrain.fixes import expm_multiply


LOGGER = logging.getLogger(__name__)


class TDVP(tt.Sweeper):
    """TDVP method for time evolution."""

    def sweep_1site_right(self, time: float) -> None:
        """Sweep from left to right, evolving 1 site at a time."""
        assert self.state.center == 0, "To sweep right we start from the left"
        assert None not in self.ham_right, "We need all right parts"
        for site in range(0, len(self.state)):
            # locally evolve the state of sites `site`
            mpo = [self.ham_left[site].copy(), self.ham[site].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            # TODO: calculate the trace for calculation of exponential
            h_eff = tt.herm_linear_operator(mpo)
            # show(mpo)
            v0 = self.state[site]
            new_node = tn.Node(
                expm_multiply(-0.5j*time*h_eff, v0.tensor.reshape(-1)).reshape(v0.shape)
            )
            if site == len(self.state) - 1:  # state evolved, stop here
                new_node.name = str(site)
                new_node.add_axis_names(tt.AXES_S)
                self.state.set_node(site, new_node)
                break

            # split state and backward evolve center of orthogonality
            left, center = tn.split_node_qr(new_node, new_node[:2], new_node[2:],
                                            left_name=f"{site}L")
            left.add_axis_names(tt.AXES_S)
            # create new left Hamiltonian for next site
            self.update_ham_left(site=site+1, state_node=left)

            mpo = [self.ham_left[site+1].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            h_eff = tt.herm_linear_operator(mpo)
            new_node = tn.Node(
                expm_multiply(+0.5j*time*h_eff, center.tensor.reshape(-1)).reshape(center.shape)
            )
            right = self.state[site+1].copy()
            tn.connect(new_node[1], right["left"])
            right = tn.contract_between(
                new_node, right, name=str(site+1),
                output_edge_order=[new_node[0], *right[1:]],
                axis_names=tt.AXES_S
            )
            self.state.set_range(site, site+2, [left, right])
            self.state.center += 1

    def sweep_1site_left(self, time: float) -> None:
        """Sweep from right to left, evolving 1 site at a time."""
        assert self.state.center == len(self.state) - 1, "To sweep left we start from the right."
        assert None not in self.ham_left, "We need all left parts"
        for site in range(len(self.state)-1, -1, -1):
            # locally evolve the state of sites `site`
            mpo = [self.ham_left[site].copy(), self.ham[site].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            # TODO: calculate the trace for calculation of exponential
            h_eff = tt.herm_linear_operator(mpo)
            # show(mpo)
            v0 = self.state[site]
            new_node = tn.Node(
                expm_multiply(-0.5j*time*h_eff, v0.tensor.reshape(-1)).reshape(v0.shape)
            )
            if site == 0:  # stated evolved, stop here
                new_node.name = str(site)
                new_node.add_axis_names(tt.AXES_S)
                self.state.set_node(site, new_node)
                break

            # split state and backward evolve center of orthogonality
            center, right = tn.split_node_rq(new_node, new_node[:1], new_node[1:],
                                             right_name=f"R{site}")
            right.add_axis_names(tt.AXES_S)
            # create new right Hamiltonian for next site
            self.update_ham_right(site=site-1, state_node=right)

            mpo = [self.ham_left[site].copy(), self.ham_right[site-1].copy()]
            tt.chain(mpo)
            h_eff = tt.herm_linear_operator(mpo)
            new_node = tn.Node(
                expm_multiply(+0.5j*time*h_eff, center.tensor.reshape(-1)).reshape(center.shape)
            )
            left = self.state[site-1].copy()
            tn.connect(left["right"], new_node[0])
            left = tn.contract_between(
                new_node, left, name=str(site-1),
                output_edge_order=[*left[:-1], new_node[1]],
                axis_names=tt.AXES_S
            )
            self.state.set_range(site-1, site+1, [left, right])
            self.state.center -= 1

    def sweep_1site(self, time: float) -> None:
        """Full TDVP sweep evolving 1 site at a time."""
        self.sweep_1site_right(time)
        self.sweep_1site_left(time)

    def sweep_2site_right(self, time: float, max_bond_dim: int, trunc_weight: float
                          ) -> List[float]:
        """Sweep from left to right, evolving two sites at a once."""
        assert self.state.center == 0, "To sweep right we start from the left"
        assert None not in self.ham_right[1:], "We need all right parts"
        # TODO: should we calculate norm here and use it form normalization?
        tws: List[float] = []
        for site in range(0, len(self.state)-1):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site].copy(), self.ham[site].copy(),
                   self.ham[site+1].copy(), self.ham_right[site+1].copy()]
            tt.chain(mpo)
            h_eff = tt.herm_linear_operator(mpo)
            node1, node2 = self.state[site:site+2]
            v0 = tn.contract_between(
                node1, node2,
                output_edge_order=[node1["left"], node1["phys"], node2["phys"], node2["right"]]
            )
            dbl_node = tn.Node(
                expm_multiply(-0.5j*time*h_eff, v0.tensor.reshape(-1)).reshape(v0.shape)
            )
            # split the tensor and compress it moving center to the right
            left, rs, rvh, trunc_s = tn.split_node_full_svd(
                dbl_node, dbl_node[:2], dbl_node[2:],
                max_singular_values=max_bond_dim, max_truncation_err=trunc_weight,
                left_name=f"{site}L",
            )
            left.add_axis_names(tt.AXES_S)
            self.update_ham_left(site+1, state_node=left)
            # keep the norm
            kept, trunc = np.sum(rs.tensor**2), np.sum(trunc_s**2)
            tws.append(np.sqrt(trunc))
            if trunc > 0:
                rs.tensor *= np.sqrt(1 + trunc/kept)
            right = tn.contract_between(
                rs, rvh, name=str(site+1), output_edge_order=[rs[0], *rvh[1:]],
                axis_names=tt.AXES_S,
            )

            if site < len(self.state) - 2:  # backward evolve center of orthogonality
                mpo = [self.ham_left[site+1].copy(),
                       self.ham[site+1].copy(),
                       self.ham_right[site+1].copy()]
                tt.chain(mpo)
                h_eff = tt.herm_linear_operator(mpo)
                right = tn.Node(
                    expm_multiply(+0.5j*time*h_eff, right.tensor.reshape(-1)).reshape(right.shape),
                    name=str(site+1), axis_names=tt.AXES_S,
                )
            self.state.set_range(site, site+2, [left, right])
            self.state.center += 1
            LOGGER.debug("Right sweep: bond-dim %3s, trunc %.3e", rs.tensor.shape[0], tws[-1])
            if tws[-1] > trunc_weight:
                LOGGER.warning("Max. bond dim %3d between sites %d--%d. Truncation error %.3e.",
                               max_bond_dim, site-1, site, tws[-1])
        return tws

    def sweep_2site_left(self, time: float, max_bond_dim: int, trunc_weight: float
                         ) -> List[float]:
        """Sweep from right to left, evolving two sites at a once."""
        assert self.state.center == len(self.state) - 1, "To sweep right we start from the left"
        assert None not in self.ham_left[:-1], "We need all left parts"
        tws: List[float] = []
        for site in range(len(self.state)-1, 0, -1):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site-1].copy(), self.ham[site-1].copy(),
                   self.ham[site].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            h_eff = tt.herm_linear_operator(mpo)
            node1, node2 = self.state[site-1:site+1]
            v0 = tn.contract_between(
                node1, node2,
                output_edge_order=[node1["left"], node1["phys"], node2["phys"], node2["right"]]
            )
            print(f"Forw-evolve {site} and {site+1}")
            dbl_node = tn.Node(
                expm_multiply(-0.5j*time*h_eff, v0.tensor.reshape(-1)).reshape(v0.shape)
            )
            # split the tensor and compress it moving center to the right
            lu, ls, right, trunc_s = tn.split_node_full_svd(
                dbl_node, dbl_node[:2], dbl_node[2:],
                max_singular_values=max_bond_dim, max_truncation_err=trunc_weight,
                right_name=f"R{site}",
            )
            right.add_axis_names(tt.AXES_S)
            self.update_ham_right(site-1, state_node=right)
            # keep the norm
            kept, trunc = np.sum(ls.tensor**2), np.sum(trunc_s**2)
            tws.append(np.sqrt(trunc))
            if trunc > 0:
                ls.tensor *= np.sqrt(1 + trunc/kept)
            left = tn.contract_between(
                lu, ls, name=str(site-1), output_edge_order=[*lu[:-1], ls[1]],
                axis_names=tt.AXES_S,
            )
            if site > 1:  # backward evolve center of orthogonality
                print(f"Back-evolve {site-1}")
                mpo = [self.ham_left[site-1].copy(),
                       self.ham[site-1].copy(),
                       self.ham_right[site-1].copy()]
                tt.chain(mpo)
                h_eff = tt.herm_linear_operator(mpo)
                left = tn.Node(
                    expm_multiply(+0.5j*time*h_eff, left.tensor.reshape(-1)).reshape(left.shape),
                    name=str(site-1), axis_names=tt.AXES_S,
                )
            self.state.set_range(site-1, site+1, [left, right])
            self.state.center -= 1
            LOGGER.debug("Left sweep: bond-dim %3s, trunc %.3e", ls.tensor.shape[0], tws[-1])
            if tws[-1] > trunc_weight:
                LOGGER.warning("Max. bond dim %3d between sites %d--%d. Truncation error %.3e.",
                               max_bond_dim, site-1, site, tws[-1])
        return tws

    def sweep_2site(self, time: float, max_bond_dim: int, trunc_weight: float):
        """Full TDVP sweep evolving 2 sites at once."""
        tw_right = self.sweep_2site_right(time, max_bond_dim, trunc_weight)
        tw_left = self.sweep_2site_left(time, max_bond_dim, trunc_weight)
        return tw_right + tw_left
