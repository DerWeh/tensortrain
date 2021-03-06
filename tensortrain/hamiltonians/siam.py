"""SIAM Hamiltonian in star geometry."""
import numpy as np
import tensornetwork as tn

from gftool import siam

import tensortrain as tt

DIM = 2  #: local dimension ordered as phys_in, phys_out
ORAISE = np.array([[0, 1],
                   [0, 0]], dtype=np.int8)
ONUMBER = np.array([[0, 0],
                    [0, 1]], dtype=np.int8)
OPARITY = np.array([[1, 0],
                    [0, -1]], dtype=np.int8)
IDX = np.eye(DIM, dtype=np.int8)

assert np.allclose(ORAISE @ ONUMBER, ORAISE)
assert np.allclose(ORAISE.T @ ONUMBER, 0)
assert np.allclose(ONUMBER @ ONUMBER, ONUMBER)
assert np.allclose(OPARITY @ OPARITY, IDX)

for const in (ORAISE, ONUMBER, OPARITY, IDX):
    const.setflags(write=False)


def siam_hamiltonain(e_onsite: float, interaction: float, e_bath, hopping) -> tt.Operator:
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
    for eps, hop in zip(e_bath[1:], hopping[1:]):
        Hi = wi.copy()
        Hi[1, :, :, :] = [eps*ONUMBER, IDX, hop*ORAISE, hop.conj()*ORAISE.T]
        Hup.append(Hi)
    Hdn = []
    Hdn.append(np.array([e_bath[-1]*ONUMBER, IDX, hopping[-1]*ORAISE,
                         hopping[-1].conj()*ORAISE.T])[:, None])
    for eps, hop in zip(e_bath[-2::-1], hopping[-2::-1]):
        Hi = wi.copy()
        Hi[:, 1, :, :] = [eps*ONUMBER, IDX, hop*ORAISE, hop.conj()*ORAISE.T]
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
    nodes = [tn.Node(hi, name=f"H{site}???", axis_names=tt.AXES_O) for site, hi in enumerate(Hup)]
    nodes.append(tn.Node(Himpup, name="Himp???", axis_names=tt.AXES_O))
    nodes.append(tn.Node(Himpdn, name="Himp???", axis_names=tt.AXES_O))
    _nodes = [tn.Node(hi, name=f"H{site}???", axis_names=tt.AXES_O) for site, hi in enumerate(Hdn)]
    nodes += _nodes[::-1]
    tt.chain([node_l] + nodes + [node_r])
    return tt.Operator(nodes, left=node_l, right=node_r)


def apply_local(site: int, ket: tt.State, operator: tn.Node,
                signs=False, canonicalize=False) -> tt.State:
    """Apply local `operator` at `site` of state `ket`.

    Parameters
    ----------
    site : int
        Site at which local operator is applied.
    ket : tt.State
        State to which the operator is applied.
    operator : tn.Node
        Local node, has to have axes two "phys_in" and "phys_out".
    signs : bool, optional
        Weather to apply fermionic signs, should be done for anti-commuting
        operators like creation and annihilation and not for commuting
        operators like the density. (default: False)
    canonicalize : bool, optional
        Weather to set site 0 as center of orthogonality. (default: False)

    Returns
    -------
    tt.State
        New state with applied operator.

    """
    if site >= len(ket) or site < 0:
        raise ValueError(f"Invalid site {site} for state of length {len(ket)}.")
    ket = ket.copy()
    if signs:
        # fermionic sign
        for pos in range(site):
            sign = tn.Node(OPARITY, axis_names=["phys_in", "phys_out"])
            node = ket[pos]
            tn.connect(node["phys"], sign["phys_in"])
            ket.nodes[pos] = tn.contract_between(
                node, sign, name=str(pos),
                output_edge_order=(node["left"], sign["phys_out"], node["right"]),
                axis_names=tt.AXES_S
            )
    # actual increase of particle number
    node = ket[site]
    tn.connect(node["phys"], operator["phys_in"])
    ket.nodes[site] = tn.contract_between(
        node, operator, name=str(site),
        output_edge_order=(node["left"], operator["phys_out"], node["right"]),
        axis_names=tt.AXES_S
    )
    if canonicalize:
        if ket.center < site:
            ket.center = site
        ket.set_center(0)
    else:
        ket.center = None
    return ket


def apply_number(site: int, ket: tt.State, canonicalize=False) -> tt.State:
    """Apply number operator at `site` to state `ket`.

    Parameters
    ----------
    site : int
        Position of the site.
    ket : tt.State
        State the operator is applied to.
    canonicalize : bool, optional
        Whether to set site ``0`` as the center of orthogonality of the new state.

    Returns
    -------
    tt.State
        The new state.

    """
    onumber = tn.Node(ONUMBER, axis_names=["phys_in", "phys_out"])
    return apply_local(site, ket, operator=onumber, signs=False, canonicalize=canonicalize)


def apply_creation(site: int, ket: tt.State, canonicalize=False) -> tt.State:
    """Apply creation operator at `site` to state `ket`.

    Parameters
    ----------
    site : int
        Position of the site.
    ket : tt.State
        State the operator is applied to.
    canonicalize : bool, optional
        Whether to set site ``0`` as the center of orthogonality of the new state.

    Returns
    -------
    tt.State
        The new state.

    """
    oraise = tn.Node(ORAISE, axis_names=["phys_in", "phys_out"])
    return apply_local(site, ket, operator=oraise, signs=True, canonicalize=canonicalize)


def apply_annihilation(site: int, ket: tt.State, canonicalize=False) -> tt.State:
    """Apply annihilation operator at `site` to state `ket`.

    Parameters
    ----------
    site : int
        Position of the site.
    ket : tt.State
        State the operator is applied to.
    canonicalize : bool, optional
        Whether to set site ``0`` as the center of orthogonality of the new state.

    Returns
    -------
    tt.State
        The new state.

    """
    oraise = tn.Node(ORAISE.T, axis_names=["phys_in", "phys_out"])
    return apply_local(site, ket, operator=oraise, signs=True, canonicalize=canonicalize)


def exact_energy(e_onsite, e_bath, hopping) -> float:
    """Exact ground state energy for non-interacting system."""
    ham = siam.hamiltonian_matrix(e_onsite, e_bath=e_bath, hopping=hopping)
    eigs = np.linalg.eigvalsh(ham)
    return 2 * np.sum(eigs[eigs < 0])  # all energies < 0 are double occupied
