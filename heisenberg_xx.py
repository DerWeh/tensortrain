"""Calculate ground-state of Heisenberg XX model using DMRG."""
import logging

import numpy as np

import tensortrain as tt

# example run
if __name__ == '__main__':
    # Set parameters
    SIZE = 50
    MAX_BOND_DIM = 70   # runtime should behave like O(bond_dim**3)
    TRUNC_WEIGHT = 1e-6
    SWEEPS = 2
    SHOW = True
    tt.setup_logging(level=logging.DEBUG)  # print all
    # tt.setup_logging(level=logging.INFO)

    # Initialize state and operator
    mps = tt.State.from_random(
        phys_dims=[2]*SIZE,
        bond_dims=[min(2**(site), 2**(SIZE-site), MAX_BOND_DIM//4)
                   for site in range(SIZE-1)]
    )
    mpo = tt.heisenbergxx.xx_hamiltonian(len(mps))
    dmrg = tt.DMRG(mps, mpo)

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
    gs_energy = tt.heisenbergxx.exact_energy(len(mps))
    print(f"Final GS energy: {energies[-1]}"
          f"\nTrue error: {(energies[-1] - gs_energy)/abs(gs_energy)}"
          f" (absolute {energies[-1] - gs_energy})"
          f" (max truncation in last iteration {max(err)})")
    if SHOW:  # plot results
        import matplotlib.pyplot as plt
        plt.plot(energies - gs_energy)
        plt.yscale("log")
        plt.ylabel(r"$E^{\mathrm{DMRG}} - E^{\mathrm{exact}}$")
        plt.show()
