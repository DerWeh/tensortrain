"""Test SIAM Hamiltonian against non-interacting result."""
import gftool as gt
import numpy as np
import pytest

from gftool import siam
from scipy.sparse.linalg import eigsh

import tensortrain as tt

from test.dmrg_test import siam0_gs

assert siam0_gs


def test_gs0_energy():
    """Compare exact diagonalization against Lanczos ground-state."""
    rng = np.random.default_rng(0)
    sites = 4
    parameters = {
        "e_onsite": np.array(0.2),
        "e_bath": 2*rng.random(sites) - 1,
        "hopping": rng.random(sites),
    }
    ham = tt.siam.siam_hamiltonain(**parameters, interaction=0)
    ham_op = tt.herm_linear_operator([ham.left] + ham.nodes + [ham.right])
    energy, __ = eigsh(ham_op, k=1, which="SA")
    energy0 = tt.siam.exact_energy(**parameters)
    assert energy.item() == pytest.approx(energy0)


def test_occupation0(siam0_gs):
    """Test non-interacting interaction, should test operators."""
    dmrg: tt.DMRG
    prm, dmrg = siam0_gs
    del prm["interaction"]
    n_bath = prm["e_bath"].size
    ham0_mat = siam.hamiltonian_matrix(**prm)
    dec = gt.matrix.decompose_her(ham0_mat)
    assert not np.any(np.isclose(dec.eig, 0)), "Eigenvalue close 0 is hard at T=0"
    occ = dec.eig < 0
    occ_imp = dec.reconstruct(occ, kind='diag')[0]

    oket = tt.siam.apply_number(n_bath, dmrg.state)
    assert tt.inner(dmrg.state, oket) == pytest.approx(occ_imp)
    cd_ket = tt.siam.apply_number(n_bath, ket=dmrg.state)
    assert tt.inner(cd_ket, cd_ket) == pytest.approx(occ_imp)


def test_operators():
    """Check basic properties of operators."""
    n_sites = 10
    ket = tt.State.from_random(
        phys_dims=[2]*n_sites,
        bond_dims=[10]*(n_sites-1),
        seed=0
    )

    # test occupation properties of operators
    n_ket = tt.siam.apply_number(n_sites//2, ket=ket)
    nn_ket = tt.siam.apply_number(n_sites//2, ket=n_ket)
    assert n_ket == nn_ket, "Fermi"
    c_ket = tt.siam.apply_annihilation(n_sites//2, ket=ket)
    cc_ket = tt.siam.apply_annihilation(n_sites//2, ket=c_ket)
    assert tt.inner(c_ket, cc_ket) == pytest.approx(0), "Fermi"
    cdc_ket = tt.siam.apply_creation(n_sites//2, ket=c_ket)
    assert tt.inner(ket, cdc_ket) == pytest.approx(tt.inner(ket, n_ket))
    assert tt.inner(c_ket, c_ket) == pytest.approx(tt.inner(ket, n_ket))

    # test canonicalization
    n_ket = tt.siam.apply_number(n_sites//2, ket=ket, canonicalize=True)
    assert np.sum(n_ket[0].tensor*n_ket[0].tensor.conj()) == pytest.approx(tt.inner(n_ket, n_ket))
    c_ket = tt.siam.apply_annihilation(n_sites//2, ket=ket, canonicalize=True)
    assert np.sum(c_ket[0].tensor*c_ket[0].tensor.conj()) == pytest.approx(tt.inner(c_ket, c_ket))
    cd_ket = tt.siam.apply_creation(n_sites//2, ket=ket, canonicalize=True)
    assert np.sum(cd_ket[0].tensor*cd_ket[0].tensor.conj()) == pytest.approx(tt.inner(cd_ket, cd_ket))
