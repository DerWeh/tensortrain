"""Solve SIAM using DMRG."""
import numpy as np

import tensortrain as tt


# example run
if __name__ == '__main__':
    # Set parameters
    import logging
    MAX_BOND_DIM = 50   # runtime should be have like O(bond_dim**3)
    TRUNC_WEIGHT = 1e-6
    SWEEPS = 3
    SHOW = True
    # setup_logging(level=logging.DEBUG)  # print all
    tt.setup_logging(level=logging.INFO)

    # Initialize state and operator
    BATH_SIZE = 50
    E_ONSITE = 0
    E_BATH = np.linspace(-2, 2, num=BATH_SIZE)
    # import gftool as gt
    # HOPPING = gt.bethe_dos(E_BATH, half_bandwidth=2)**0.5
    HOPPING = np.ones(BATH_SIZE)
    U = 0
    ham = tt.siam.siam_hamiltonain(E_ONSITE, interaction=U, e_bath=E_BATH, hopping=HOPPING)
    mps = tt.State.from_random(
        phys_dims=[2]*len(ham),
        bond_dims=[min(2**(site), 2**(len(ham)-site), MAX_BOND_DIM//4)
                   for site in range(len(ham)-1)]
    )
    dmrg = tt.DMRG(mps, ham)

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
    gs_energy = tt.siam.exact_energy(np.array(E_ONSITE), e_bath=E_BATH, hopping=HOPPING)
    if U == 0:
        print(f"Final GS energy: {energies[-1]}"
              f"\nTrue error: {(energies[-1] - gs_energy)/abs(gs_energy)}"
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
