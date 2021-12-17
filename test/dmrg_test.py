"""Basic tests for DMRG."""
from typing import Tuple
import numpy as np
import pytest

from numpy.testing import assert_array_max_ulp

import tensortrain as tt


@pytest.fixture(scope="session")
def siam0_gs() -> Tuple[dict, tt.DMRG]:
    """Calculate ground-state of SIAM using DMRG."""
    max_bond_dim = 50   # runtime should be have like O(bond_dim**3)
    trunc_weight = 1e-10
    sweeps = 2

    bath_size = 5
    parameter = {
        "e_onsite": np.array(0),
        "e_bath": np.linspace(-2, 2, num=bath_size),
        "hopping": np.ones(bath_size),
        "interaction": 0,
    }

    # Initialize state and operator
    ham = tt.siam.siam_hamiltonain(**parameter)
    mps = tt.State.from_random(
        phys_dims=[2]*len(ham),
        bond_dims=[min(2**(site), 2**(len(ham)-site), max_bond_dim//4)
                   for site in range(len(ham)-1)]
    )
    dmrg = tt.DMRG(mps, ham)

    # Run DMRG
    for __ in range(sweeps):
        eng, __ = dmrg.sweep_2site(max_bond_dim, trunc_weight)
    return parameter, dmrg


def test_siam_gs_energy(siam0_gs):
    """Test that the Ground state energy is correct for the SIAM.

    For small system sizes the energy should be exact.
    """
    prm, dmrg = siam0_gs
    del prm["interaction"]

    gs_energy = tt.siam.exact_energy(**prm)
    assert_array_max_ulp(dmrg.energy, gs_energy, maxulp=10)
