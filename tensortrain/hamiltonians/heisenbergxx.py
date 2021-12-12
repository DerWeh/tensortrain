"""Homogeneous Heisenberg XX Hamiltonian."""
import numpy as np
import tensornetwork as tn

import tensortrain.basics as tt


def xx_hamiltonian(size: int) -> tt.Operator:
    """Create MPO for XX Hamiltonian."""
    phys_dim = 2
    # Pauli matrices
    # s_x = np.array([[0, 1],
    #                 [1, 0]])
    # s_y = np.array([[0, -1j],
    #                 [1j, 0]])
    # s_z = np.array([[1, 0],
    #                 [0, -1]])
    s_p = np.sqrt(2) * np.array([[0, 0],
                                [1, 0]])
    s_m = s_p.T
    idx = np.eye(2)
    mat = np.zeros([4, 4, phys_dim, phys_dim])
    mat[0, :-1, :, :] = idx, s_m, s_p
    mat[1:, -1, :, :] = s_p, s_m, idx
    left = np.array([1, 0, 0, 0]).reshape([4, 1, 1])  # left MPO boundary
    right = np.array([0, 0, 0, 1]).reshape([4, 1, 1])  # right MPO boundary
    nodes = [tn.Node(mat, name=f"H{site}", axis_names=tt.AXES_O) for site in range(size)]
    node_l = tn.Node(left, name="LH", axis_names=["right", "phys_in", "phys_out"])
    node_r = tn.Node(right, name="HR", axis_names=["left", "phys_in", "phys_out"])
    tt.chain([node_l] + nodes + [node_r])
    return tt.Operator(nodes, left=node_l, right=node_r)


def exact_energy(nsite: int) -> float:
    """Exact ground state energy computed from free fermions."""
    ham = np.diag(np.ones(nsite - 1), k=1) + np.diag(np.ones(nsite - 1), k=-1)
    eigs = np.linalg.eigvalsh(ham)
    return 2 * sum(eigs[eigs < 0])
