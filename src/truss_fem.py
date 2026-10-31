"""Minimal 2D linear truss FEM solver and verification examples."""
import numpy as np


def solve_truss(nodes, elements, E, area, loads, fixed_dofs):
    """Return displacements, reactions, axial forces and compliance in SI units.

    nodes: (n,2); elements: (m,2) zero-based node indices;
    loads: (2*n,) global nodal forces; fixed_dofs: restrained DOF indices.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    forces = np.asarray(loads, dtype=float)
    n = len(nodes)
    K = np.zeros((2*n, 2*n))
    areas = np.broadcast_to(np.asarray(area, dtype=float), (len(elements),))
    for e, (i, j) in enumerate(elements):
        d = nodes[j] - nodes[i]
        length = np.linalg.norm(d)
        if length <= 0:
            raise ValueError('Zero-length element')
        c, s = d / length
        a = np.array([-c, -s, c, s])
        ke = E*areas[e]/length * np.outer(a, a)
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        K[np.ix_(dofs, dofs)] += ke
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    free = np.setdiff1d(np.arange(2*n), fixed)
    if np.linalg.matrix_rank(K[np.ix_(free, free)], tol=1e-5) < len(free):
        raise ValueError('Mechanism or disconnected structural DOF: singular stiffness matrix')
    u = np.zeros(2*n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], forces[free])
    reactions = K @ u - forces
    axial = np.empty(len(elements))
    for e, (i, j) in enumerate(elements):
        d = nodes[j] - nodes[i]
        length = np.linalg.norm(d)
        c, s = d / length
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        axial[e] = E*areas[e]/length * (np.array([-c,-s,c,s]) @ u[dofs])
    return {'u':u, 'reactions':reactions, 'axial_forces':axial,
            'compliance':float(forces @ u), 'K':K}


def verify():
    E, A, L, F = 70e9, 100e-6, 1., 1000.
    # Test 1: axial bar, x DOF of loaded node free, both y DOFs constrained.
    sol = solve_truss([[0,0],[L,0]], [[0,1]], E,A, [0,0,F,0], [0,1,3])
    expected = F*L/(E*A)
    assert np.isclose(sol['u'][2], expected, rtol=1e-10)
    assert np.isclose(sol['axial_forces'][0], F)
    assert np.isclose(sol['reactions'][0], -F)
    # Test 2: two diagonal members forming a symmetric V (both bases fixed).
    nodes = [[0,0],[2,0],[1,1]]
    sol2=solve_truss(nodes, [[0,2],[1,2]], E,A, [0,0,0,0,0,-F], [0,1,2,3])
    # Each member supports F/sqrt(2), extension F*Lmember/(sqrt(2)*EA).
    # vertical tip displacement = F*sqrt(2)/(EA).
    expected_v = -F*np.sqrt(2)/(E*A)
    assert np.isclose(sol2['u'][5], expected_v, rtol=1e-10)
    assert np.isclose(sol2['u'][4], 0, atol=1e-12)
    assert np.allclose(sol2['axial_forces'], [-F/np.sqrt(2)]*2)
    # Test 3: deliberate mechanism (free vertical DOF at end of horizontal bar).
    try:
        solve_truss([[0,0],[1,0]], [[0,1]], E,A, [0,0,F,0], [0,1])
    except ValueError:
        pass
    else:
        raise AssertionError('Mechanism not detected')
    print('PASS: axial bar / analytical tip displacement:', expected)
    print('PASS: symmetric two-member V / analytical tip displacement:', expected_v)
    print('PASS: singular mechanism rejected')

if __name__ == '__main__':
    verify()
