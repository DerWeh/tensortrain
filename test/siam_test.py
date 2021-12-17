"""Test SIAM Hamiltonian against non-interacting result."""
import numpy as np
import pytest

from scipy.sparse.linalg import eigsh
from gftool import siam

import tensortrain as tt


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
