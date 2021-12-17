"""Very crude tests for some basic functionality."""
import numpy as np
import pytest

import tensortrain as tt


def test_norm():
    """Random states should be normalized."""
    n_sites = 10
    bra = tt.State.from_random(
        phys_dims=[2]*n_sites,
        bond_dims=[10]*(n_sites-1),
    )
    assert tt.inner(bra, bra) == pytest.approx(1)


def test_equality():
    """Check if equality operator is implemented."""
    n_sites = 10
    bra = tt.State.from_random(
        phys_dims=[2]*n_sites,
        bond_dims=[10]*(n_sites-1),
    )
    ket = bra.copy(conjugate=True)
    assert bra == ket, "Random state is real."
    ket[0].tensor += 1
    assert bra != ket


def test_operator_trace():
    """Test trace for simple example."""
    rng = np.random.default_rng(0)
    sites = 4
    parameters = {
        "e_onsite": np.array(0.2),
        "e_bath": 2*rng.random(sites) - 1,
        "hopping": rng.random(sites),
    }
    ham = tt.siam.siam_hamiltonain(**parameters, interaction=0)
    ham_op = tt.herm_linear_operator([ham.left] + ham.nodes + [ham.right])
    trace = tt.basics.trace([ham.left] + ham.nodes + [ham.right])
    dense = ham_op @ np.eye(*ham_op.shape)
    assert trace == pytest.approx(np.trace(dense))
