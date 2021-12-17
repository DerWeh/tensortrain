"""Basic test for TDVP."""
import numpy as np

from numpy.testing import assert_allclose

import tensortrain as tt
from test.dmrg_test import siam0_gs

assert siam0_gs


def test_tdvp1_siam_gs_evolution(siam0_gs):
    """Test single-site TDVP for the ground state of the SIAM.

    For the ground state, the time evolution should be just the GS energy.
    """
    dmrg: tt.DMRG
    __, dmrg = siam0_gs

    # TODO check that norm is conserved
    gs_energy = dmrg.energy
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


def test_tdvp2_siam_gs_evolution(siam0_gs):
    """Test two-site TDVP for the ground state of the SIAM.

    For the ground state, the time evolution should be just the GS energy.
    """
    dmrg: tt.DMRG
    __, dmrg = siam0_gs

    gs_energy = dmrg.energy
    tevo = tt.TDVP(state=dmrg.state, ham=dmrg.ham)
    time_step = 0.1
    tevo.sweep_2site_right(time_step, max_bond_dim=50, trunc_weight=1e-8)
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
