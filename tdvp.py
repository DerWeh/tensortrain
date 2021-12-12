"""Time-depended variational principle for tensor trains."""
import logging

from typing import List

import numpy as np
import tensornetwork as tn

import siam

import tensortrain as tt

from tensortrain.fixes import expm_multiply
from dmrg_tn import setup_logging, DMRG


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


# example run
if __name__ == '__main__':
    # Set parameters
    import logging
    MAX_BOND_DIM = 50   # runtime should be have like O(bond_dim**3)
    TRUNC_WEIGHT = 1e-10
    SWEEPS = 2
    SHOW = True
    # setup_logging(level=logging.DEBUG)  # print all
    setup_logging(level=logging.INFO)

    # Initialize state and operator
    BATH_SIZE = 10
    E_ONSITE = 0
    E_BATH = np.linspace(-2, 2, num=BATH_SIZE)
    # import gftool as gt
    # HOPPING = gt.bethe_dos(E_BATH, half_bandwidth=2)**0.5
    HOPPING = np.ones(BATH_SIZE)
    U = 0
    ham = siam.siam_mpo(E_ONSITE, interaction=U, e_bath=E_BATH, hopping=HOPPING)
    mps = tt.State.from_random(
        phys_dims=[2]*len(ham),
        bond_dims=[min(2**(site), 2**(len(ham)-site), MAX_BOND_DIM//4)
                   for site in range(len(ham)-1)]
    )
    dmrg = DMRG(mps, ham)

    # Run DMRG
    energies, errors = [], []
    for num in range(SWEEPS):
        print(f"Running sweep {num}", flush=True)
        eng, err = dmrg.sweep_2site(MAX_BOND_DIM, TRUNC_WEIGHT)
        energies += eng
        errors += err
        print(f"GS energy: {energies[-1]}, (max truncation {max(err)})", flush=True)
        H2 = dmrg.eval_ham2()
        print(f"Estimated error {abs((H2 - eng[-1]**2)/eng[-1])}", flush=True)
    energies = np.array(energies)
    gs_energy = siam.exact_energy(np.array(E_ONSITE), e_bath=E_BATH, hopping=HOPPING)
    if U == 0:
        print(f"Final GS energy: {energies[-1]}"
              f"\n True error: {(energies[-1] - gs_energy)/abs(gs_energy)}"
              f" (absolute {energies[-1] - gs_energy})"
              f" (max truncation in last iteration {max(err)})")
    if SHOW:  # plot results
        import matplotlib.pyplot as plt
        if U == 0:
            plt.plot(energies - gs_energy)
            plt.yscale("log")
            plt.ylabel(r"$E^{\mathrm{DMRG}} - E^{\mathrm{exact}}$")
            plt.tight_layout()
        #
        bond_dims = [node['right'].dimension for node in dmrg.state]
        plt.figure('bond dimensions')
        plt.axvline(len(mps)//2-0.5, color='black')
        plt.plot(bond_dims, drawstyle='steps-post')
        plt.show()