"""Basic tests for DMRG."""
import numpy as np

from numpy.testing import assert_array_max_ulp

import tensortrain as tt


def test_siam_gs_energy():
    """Test that the Ground state energy is correct for the SIAM.

    For small system sizes the energy should be exact.
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
    gs_energy = tt.siam.exact_energy(np.array(e_onsite), e_bath=e_bath, hopping=hopping)
    assert_array_max_ulp(eng[-1], gs_energy, maxulp=10)
