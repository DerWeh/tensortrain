"""Time-depended variational principle for tensor trains."""
import logging

from typing import List

import tensornetwork as tn

import tensortrain.basics as tt

from tensortrain.fixes import expm_multiply


LOGGER = logging.getLogger(__name__)


class TDVP(tt.Sweeper):
    """TDVP method for time evolution."""

    def sweep_1site_right(self, time: float) -> None:
        """Sweep from left to right, evolving one site at a time."""
        assert self.state.center == 0, "To sweep right we start from the left"
        assert None not in self.ham_right[1:], "We need all right parts"
        for site in range(0, len(self.state)):
            # locally optimize the state of sites `site` and `site+1`
            mpo = [self.ham_left[site].copy(), self.ham[site].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            # TODO: calculate the trace for calculation of exponential
            h_eff = tt.herm_linear_operator(mpo)
            # show(mpo)
            v0 = self.state[site]
            new_node = tn.Node(
                expm_multiply(-0.5j*time*h_eff, v0.tensor.reshape(-1)).reshape(v0.shape)
            )
            if site == len(self.state) - 1:  # stated evolved, stop here
                new_node.name = str(site)
                new_node.add_axis_names(tt.AXES_S)
                self.state.set_node(site, new_node)
                break

            # split state and backward evolve center of orthogonality
            left, rvh = tn.split_node_qr(new_node, new_node[:2], new_node[2:],
                                         left_name=f"{site}L")
            left.add_axis_names(tt.AXES_S)
            # create new left Hamiltonian
            mpol: List[tn.Node] = [self.ham_left[site].copy(), left.copy(),
                                   left.copy(conjugate=True), self.ham[site].copy()]
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
            mpo = [self.ham_left[site+1].copy(), self.ham_right[site].copy()]
            tt.chain(mpo)
            h_eff = tt.herm_linear_operator(mpo)
            new_node = tn.Node(
                expm_multiply(+0.5j*time*h_eff, rvh.tensor.reshape(-1)).reshape(rvh.shape)
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
