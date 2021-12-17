"""Very crude tests for some basic functionality."""
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
