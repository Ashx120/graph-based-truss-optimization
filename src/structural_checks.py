"""Preliminary axial stress and idealized Euler member buckling checks."""
import numpy as np
from src.truss_fem import solve_truss


def assess_members(nodes, elements, E, A, loads, fixed_dofs):
    result = solve_truss(nodes, elements, E, A, loads, fixed_dofs)
    axial = np.asarray(result['axial_forces'], dtype=float)
    lengths = np.linalg.norm(
        nodes[elements[:, 1]] - nodes[elements[:, 0]], axis=1
    )
    radius = np.sqrt(A / np.pi)
    I = np.pi * radius**4 / 4
    stress = axial / A
    effective_length_factor = 1.0
    critical_load = np.pi**2 * E * I / (effective_length_factor * lengths)**2
    buckling_factor = np.full(len(elements), np.inf)
    compressed = axial < -1e-9
    buckling_factor[compressed] = critical_load[compressed] / np.abs(axial[compressed])
    return {
        'axial_forces': axial,
        'stress': stress,
        'lengths': lengths,
        'radius': radius,
        'I': I,
        'buckling_factor': buckling_factor,
        'minimum_buckling_factor': float(np.min(buckling_factor)),
        'maximum_tensile_stress': float(np.max(stress)),
        'maximum_compressive_stress': float(np.min(stress)),
    }
