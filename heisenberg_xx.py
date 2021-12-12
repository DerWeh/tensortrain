"""Apply DMRG to the spin 1/2 Heisberg XX model."""
import logging

import numpy as np
import tensornetwork as tn

import tensortrain as tt
from dmrg_tn import DMRG, setup_logging


def xx_mpo(size: int) -> tt.MPO:
    """Create MPO for XX Hamiltonian."""
    phys_dim = 2
    # Pauli matrices
    # s_x = np.array([[0, 1],
    #                 [1, 0]])
    # s_y = np.array([[0, -1j],
    #                 [1j, 0]])
    # s_z = np.array([[1, 0],
    #                 [0, -1]])
    s_p = np.sqrt(2) * np.array([[0, 0],
                                [1, 0]])
    s_m = s_p.T
    idx = np.eye(2)
    mat = np.zeros([4, 4, phys_dim, phys_dim])
    mat[0, :-1, :, :] = idx, s_m, s_p
    mat[1:, -1, :, :] = s_p, s_m, idx
    left = np.array([1, 0, 0, 0]).reshape([4, 1, 1])  # left MPO boundary
    right = np.array([0, 0, 0, 1]).reshape([4, 1, 1])  # right MPO boundary
    nodes = [tn.Node(mat, name=f"H{site}", axis_names=tt.MO_AXES) for site in range(size)]
    node_l = tn.Node(left, name="LH", axis_names=["right", "phys_in", "phys_out"])
    node_r = tn.Node(right, name="HR", axis_names=["left", "phys_in", "phys_out"])
    tt.chain([node_l] + nodes + [node_r])
    return tt.MPO(nodes, left=node_l, right=node_r)


def exact_energy(nsite: int) -> float:
    """Exact ground state energy computed from free fermions."""
    ham = np.diag(np.ones(nsite - 1), k=1) + np.diag(np.ones(nsite - 1), k=-1)
    eigs = np.linalg.eigvalsh(ham)
    return 2 * sum(eigs[eigs < 0])


# example run
if __name__ == '__main__':
    # Set parameters
    SIZE = 50
    MAX_BOND_DIM = 70   # runtime should behave like O(bond_dim**3)
    TRUNC_WEIGHT = 1e-6
    SWEEPS = 2
    SHOW = True
    setup_logging(level=logging.DEBUG)  # print all
    # setup_logging(level=logging.INFO)

    # Initialize state and operator
    mps = tt.MPS.from_random(
        phys_dims=[2]*SIZE,
        bond_dims=[min(2**(site), 2**(SIZE-site), MAX_BOND_DIM//4)
                   for site in range(SIZE-1)]
    )
    mpo = xx_mpo(len(mps))
    dmrg = DMRG(mps, mpo)

    # Run DMRG
    energies, errors = [], []
    for num in range(SWEEPS):
        print(f"Running sweep {num}", flush=True)
        eng, err = dmrg.sweep_2site(MAX_BOND_DIM, TRUNC_WEIGHT)
        energies += eng
        errors += err
        H2 = dmrg.eval_ham2()
        print(f"GS energy: {energies[-1]}, (max truncation {max(err)})", flush=True)
        print(f"Estimated error {abs((H2 - eng[-1]**2)/eng[-1])}", flush=True)
    energies = np.array(energies)
    gs_energy = exact_energy(len(mps))
    print(f"Final GS energy: {energies[-1]}"
          f"\n True error: {(energies[-1] - gs_energy)/abs(gs_energy)}"
          f" (absolute {energies[-1] - gs_energy})"
          f" (max truncation in last iteration {max(err)})")
    if SHOW:  # plot results
        import matplotlib.pyplot as plt
        plt.plot(energies - gs_energy)
        plt.yscale("log")
        plt.ylabel(r"$E^{\mathrm{DMRG}} - E^{\mathrm{exact}}$")
        plt.show()
