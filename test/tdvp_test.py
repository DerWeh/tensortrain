"""Basic test for TDVP."""
import numpy as np

from numpy.testing import assert_allclose

import tensortrain as tt
import siam


def test_tdvp1_siam_gs_evolution():
    """Test that the Ground state energy is correct for the SIAM.

    For the ground state, the time evolution should be just the GS energy.
    """
    max_bond_dim = 50   # runtime should be have like O(bond_dim**3)
    trunc_weight = 1e-10
    sweeps = 2

    # Initialize state and operator
    bath_size = 5
    e_onsite = 0
    e_bath = np.linspace(-2, 2, num=bath_size)
    hopping = np.ones(bath_size)
    ham = siam.siam_mpo(e_onsite, interaction=0, e_bath=e_bath, hopping=hopping)
    mps = tt.State.from_random(
        phys_dims=[2]*len(ham),
        bond_dims=[min(2**(site), 2**(len(ham)-site), max_bond_dim//4)
                   for site in range(len(ham)-1)]
    )
    dmrg = tt.DMRG(mps, ham)

    # Run DMRG
    for __ in range(sweeps):
        eng, __ = dmrg.sweep_2site(max_bond_dim, trunc_weight)

    gs_energy = eng[-1]
    tevo = tt.TDVP(state=dmrg.state, ham=dmrg.ham)
    time_step = 0.1
    tevo.sweep_1site_right(time_step)
    # one sweep is half a time evolution
    overlap = tt.inner(dmrg.state, tevo.state)
    assert_allclose(overlap, np.exp(-0.5j*time_step*gs_energy), rtol=1e-14)
