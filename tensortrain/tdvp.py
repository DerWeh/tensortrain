"""Time-depended variational principle for tensor trains."""
import logging

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
            if site == len(self.state) - 1:  # stated evolved, stop here
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
