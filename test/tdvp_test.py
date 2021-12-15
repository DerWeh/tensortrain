"""Basic test for TDVP."""
import numpy as np

from numpy.testing import assert_allclose

import tensortrain as tt


def test_tdvp1_siam_gs_evolution():
    """Test single-site TDVP for the ground state of the SIAM.

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
    ham = tt.siam.siam_hamiltonain(e_onsite, interaction=0, e_bath=e_bath, hopping=hopping)
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
    tevo.sweep_1site_left(time_step)
    # another half step
    overlap = tt.inner(dmrg.state, tevo.state)
    assert_allclose(overlap, np.exp(-1.0j*time_step*gs_energy), rtol=1e-14)
    # do a full sweep
    tevo.sweep_1site(time_step)
    overlap = tt.inner(dmrg.state, tevo.state)
    assert_allclose(overlap, np.exp(-2.0j*time_step*gs_energy), rtol=1e-14)


def test_tdvp2_siam_gs_evolution():
    """Test two-site TDVP for the ground state of the SIAM.

    For the ground state, the time evolution should be just the GS energy.
    """
    max_bond_dim = 50   # runtime should be have like O(bond_dim**3)
    trunc_weight = 1e-10
    sweeps = 3

    # Initialize state and operator
    bath_size = 10  # for some reason it is bad for small systems
    e_onsite = 0
    e_bath = np.linspace(-2, 2, num=bath_size)
    hopping = np.ones(bath_size)
    ham = tt.siam.siam_hamiltonain(e_onsite, interaction=0, e_bath=e_bath, hopping=hopping)
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
    tevo.sweep_2site_right(time_step, max_bond_dim=max_bond_dim, trunc_weight=1e-8)
    # one sweep is half a time evolution
    overlap = tt.inner(dmrg.state, tevo.state)
    assert_allclose(overlap, np.exp(-0.5j*time_step*gs_energy), rtol=1e-12)
    tevo.sweep_1site_left(time_step)
    # another half step
    overlap = tt.inner(dmrg.state, tevo.state)
    assert_allclose(overlap, np.exp(-1.0j*time_step*gs_energy), rtol=1e-12)
    # do a full sweep
    tevo.sweep_1site(time_step)
    overlap = tt.inner(dmrg.state, tevo.state)
    assert_allclose(overlap, np.exp(-2.0j*time_step*gs_energy), rtol=1e-12)
