"""Solve SIAM using DMRG."""
import numpy as np
import tensornetwork as tn

from gftool import siam

from dmrg_tn import MPS, MPO, DMRG, chain, MO_AXES, setup_logging

DIM = 2  #: local dimension
ORAISE = np.array([[0, 1],
                   [0, 0]], dtype=np.int8)
ONUMBER = np.array([[0, 0],
                    [0, 1]], dtype=np.int8)
OPARITY = np.array([[1, 0],
                    [0, -1]], dtype=np.int8)
IDX = np.eye(DIM, dtype=np.int8)

for const in (ORAISE, ONUMBER, OPARITY, IDX):
    const.setflags(write=False)


def siam_mpo(e_onsite: float, interaction: float, e_bath, hopping) -> MPO:
    """Construct MPO for the SIAM in star geometry.

    Currently no spin-dependent parameters allowed, up/dn spin are identically.
    The order of bath sites can affect the necessary bond dimensions.

    Parameters
    ----------
    e_onsite : float
        On-site energy of the impurity site.
    interaction : float
        Local on-site interaction (Hubbard U).
    e_bath : (Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.

    Returns
    -------
    MPO
        Matrix product operator of the SIAM Hamiltonian.

    """
    e_bath, hopping = np.asarray(e_bath), np.asarray(hopping)
    if e_bath.shape[-1] != hopping.shape[-1]:
        raise ValueError
    # diagonal skeleton
    wi = np.zeros([4, 4, DIM, DIM])
    wi[0, 0, :, :] = wi[1, 1, :, :] = IDX
    wi[2, 2, :, :] = wi[3, 3, :, :] = OPARITY
    Hup = []
    Hup.append(np.array([e_bath[0]*ONUMBER, IDX, hopping[0]*ORAISE,
                         hopping[0].conj()*ORAISE.T])[None, :])
    for eps, tt in zip(e_bath[1:], hopping[1:]):
        Hi = wi.copy()
        Hi[1, :, :, :] = [eps*ONUMBER, IDX, tt*ORAISE, tt.conj()*ORAISE.T]
        Hup.append(Hi)
    Hdn = []
    Hdn.append(np.array([e_bath[-1]*ONUMBER, IDX, hopping[-1]*ORAISE,
                         hopping[-1].conj()*ORAISE.T])[:, None])
    for eps, tt in zip(e_bath[-2::-1], hopping[-2::-1]):
        Hi = wi.copy()
        Hi[:, 1, :, :] = [eps*ONUMBER, IDX, tt*ORAISE, tt.conj()*ORAISE.T]
        Hdn.append(Hi)
    Himpup = np.zeros([4, 3, DIM, DIM])
    Himpup[1, :, :, :] = [IDX, e_onsite*ONUMBER, ONUMBER]
    Himpup[:, 1, :, :] = [IDX, e_onsite*ONUMBER, ORAISE.T, ORAISE]
    Himpdn = np.zeros([3, 4, DIM, DIM])
    Himpdn[0, :, :, :] = [IDX, e_onsite*ONUMBER, ORAISE.T, ORAISE]
    Himpdn[:, 1, :, :] = [e_onsite*ONUMBER, IDX, interaction*ONUMBER]
    end = np.ones([1, 1, 1], dtype=np.int8)
    node_l = tn.Node(end, name="LH", axis_names=["right", "phys_in", "phys_out"])
    node_r = tn.Node(end, name="HR", axis_names=["left", "phys_in", "phys_out"])
    nodes = [tn.Node(hi, name=f"H{site}↑", axis_names=MO_AXES) for site, hi in enumerate(Hup)]
    nodes.append(tn.Node(Himpup, name="Himp↑", axis_names=MO_AXES))
    nodes.append(tn.Node(Himpdn, name="Himp↓", axis_names=MO_AXES))
    _nodes = [tn.Node(hi, name=f"H{site}↓", axis_names=MO_AXES) for site, hi in enumerate(Hdn)]
    nodes += _nodes[::-1]
    chain([node_l] + nodes + [node_r])
    return MPO(nodes, left=node_l, right=node_r)


def exact_energy(e_onsite, e_bath, hopping) -> float:
    """Exact ground state energy computed from free fermions."""
    ham = siam.hamiltonian_matrix(e_onsite, e_bath=e_bath, hopping=hopping)
    eigs = np.linalg.eigvalsh(ham)
    return 2 * np.sum(eigs[eigs < 0])  # all energies < 0 are double occupied


# example run
if __name__ == '__main__':
    # Set parameters
    import logging
    MAX_BOND_DIM = 50   # runtime should be have like O(bond_dim**3)
    TRUNC_WEIGHT = 1e-6
    SWEEPS = 3
    SHOW = True
    setup_logging(level=logging.DEBUG)  # print all
    # setup_logging(level=logging.INFO)

    # Initialize state and operator
    BATH_SIZE = 50
    E_ONSITE = 0
    E_BATH = np.linspace(-2, 2, num=BATH_SIZE)
    # import gftool as gt
    # HOPPING = gt.bethe_dos(E_BATH, half_bandwidth=2)**0.5
    HOPPING = np.ones(BATH_SIZE)
    U = 0
    ham = siam_mpo(E_ONSITE, interaction=U, e_bath=E_BATH, hopping=HOPPING)
    mps = MPS.from_random(
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
        H2 = dmrg.eval_ham2()
        print(f"GS energy: {energies[-1]}, (max truncation {max(err)})", flush=True)
        print(f"Estimated error {abs((H2 - eng[-1]**2)/eng[-1])}", flush=True)
    energies = np.array(energies)
    gs_energy = exact_energy(np.array(E_ONSITE), e_bath=E_BATH, hopping=HOPPING)
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
        bond_dims = [node['right'].dimension for node in dmrg.mps]
        plt.figure('bond dimensions')
        plt.axvline(len(mps)//2-0.5, color='black')
        plt.plot(bond_dims, drawstyle='steps-post')
        plt.show()
