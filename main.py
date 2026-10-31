"""2D truss ground-structure analysis and greedy topology optimization.

Run from the project root: python main.py
Requires src/truss_fem.py, src/graph_optimizer.py and src/structural_checks.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.truss_fem import solve_truss
from src.structural_checks import assess_members
from src.graph_optimizer import (
    compliance_constrained_optimize,
    force_based_optimize,
)

# 1. Material and geometry (SI units)
E = 70e9             # Pa
A = 100e-6           # m^2
rho = 2700           # kg/m^3
L = 1.0              # m
H = 0.3              # m
F = 1000.0           # N
nx, ny = 9, 3        # Grid columns and rows

# 2. Nodes and ground-structure connectivity
# Node index = row * nx + column; bottom-right node is nx - 1.
nodes = np.array([
    [x, y]
    for y in np.linspace(0.0, H, ny)
    for x in np.linspace(0.0, L, nx)
], dtype=float)

edges = []
for row in range(ny):
    for col in range(nx):
        i = row * nx + col
        if col < nx - 1:
            edges.append((i, i + 1))  # Horizontal
        if row < ny - 1:
            edges.append((i, i + nx))  # Vertical
        if col < nx - 1 and row < ny - 1:
            edges.append((i, i + nx + 1))  # Rising diagonal
            edges.append((i + 1, i + nx))  # Falling diagonal

elements = np.asarray(edges, dtype=int)

# 3. Graph adjacency matrix
adjacency = np.zeros((len(nodes), len(nodes)), dtype=int)
for i, j in elements:
    adjacency[i, j] = 1
    adjacency[j, i] = 1
assert np.array_equal(adjacency, adjacency.T)
assert np.all(np.diag(adjacency) == 0)
assert adjacency.sum() // 2 == len(elements)
print(f"Adjacency shape: {adjacency.shape}; graph edges: {len(elements)}")

# 4. Boundary conditions: all 3 nodes on the left edge fixed
loads = np.zeros(2 * len(nodes))
tip_node = nx - 1  # Bottom-right node
loads[2 * tip_node + 1] = -F
left_nodes = [row * nx for row in range(ny)]
fixed_dofs = [dof for node in left_nodes for dof in (2 * node, 2 * node + 1)]

# 5. Baseline FEM solution
result = solve_truss(nodes, elements, E, A, loads, fixed_dofs)
u = result["u"].reshape(-1, 2)
axial = result["axial_forces"]
compliance = result["compliance"]
lengths = np.linalg.norm(nodes[elements[:, 1]] - nodes[elements[:, 0]], axis=1)
mass = rho * A * lengths.sum()

print("\n--- Baseline FEM results ---")
print(f"Nodes: {len(nodes)}; members: {len(elements)}")
print(f"Tip vertical displacement: {u[tip_node, 1] * 1e3:.6f} mm")
print(f"Compliance: {compliance:.6f} J")
print(f"Structural mass: {mass:.6f} kg")
print(f"Maximum absolute axial force: {np.max(np.abs(axial)):.2f} N")

# Optional: detailed member forces; switch to True to print all 74 rows
PRINT_MEMBER_FORCES = False
if PRINT_MEMBER_FORCES:
    print("\n--- Initial member axial forces ---")
    for (i, j), force in zip(elements, axial):
        state = "Tension" if force > 0 else "Compression" if force < 0 else "Zero"
        print(f"({i}, {j}): {force:9.2f} N [{state}]")

# 6. Verify strain energy independently via axial extensions
strain_energy = 0.0
internal_forces = np.zeros_like(loads)
for k, (i, j) in enumerate(elements):
    direction = nodes[j] - nodes[i]
    Le = np.linalg.norm(direction)
    n = direction / Le
    extension = np.dot(u[j] - u[i], n)
    strain_energy += 0.5 * E * A / Le * extension**2

    fe = axial[k] * np.array([-n[0], -n[1], n[0], n[1]])
    dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    internal_forces[dofs] += fe

external_energy = 0.5 * np.dot(loads, result["u"])
assert np.isclose(strain_energy, external_energy, rtol=1e-8, atol=1e-10)

# 7. Verify force equilibrium, including support reactions
reactions = internal_forces - loads
free_dofs = np.setdiff1d(np.arange(len(loads)), fixed_dofs)
residual = np.max(np.abs(reactions[free_dofs]))
rx = sum(reactions[2 * node] for node in left_nodes)
ry = sum(reactions[2 * node + 1] for node in left_nodes)
assert np.allclose(reactions[free_dofs], 0, atol=1e-6)
assert np.isclose(rx, 0.0, atol=1e-6)
assert np.isclose(ry, F, atol=1e-6)

print("\n--- FEM consistency checks ---")
print(f"Element strain energy: {strain_energy:.6f} J")
print(f"External work energy: {external_energy:.6f} J")
print(f"Maximum free-DOF force residual: {residual:.3e} N")
print(f"Sum of support reactions (X): {rx:.6f} N")
print(f"Sum of support reactions (Y): {ry:.6f} N")

# 8. Plot baseline deformation (amplified for visibility)
scale = 30.0
changed_nodes = nodes + scale * u
fig, ax = plt.subplots(figsize=(12, 4.5))
for i, j in elements:
    ax.plot(nodes[[i, j], 0], nodes[[i, j], 1], "k--", alpha=0.25, lw=0.9)
    ax.plot(changed_nodes[[i, j], 0], changed_nodes[[i, j], 1],
            color="tab:blue", lw=1.4)
ax.scatter(nodes[:, 0], nodes[:, 1], color="black", s=12, zorder=3)
ax.scatter(changed_nodes[:, 0], changed_nodes[:, 1], color="tab:blue", s=12, zorder=3)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"Baseline FEM deformation ({scale:g}x visual amplification)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("cantilever_deformation.png", dpi=200)
plt.close(fig)

# 9. Compliance-constrained topology optimization
max_compliance_increase = 0.05
compliance_limit = compliance * (1 + max_compliance_increase)
print(f"\nCompliance limit: {compliance_limit:.6f} J ")

optimized_elements, history = compliance_constrained_optimize(
    nodes=nodes, elements=elements, E=E, A=A, rho=rho,
    loads=loads, fixed_dofs=fixed_dofs,
    max_compliance_increase=max_compliance_increase,
)

initial_edges = {tuple(sorted(edge)) for edge in elements}
final_edges = {tuple(sorted(edge)) for edge in optimized_elements}
removed_edges = sorted(initial_edges - final_edges)
print("\n--- Compliance-based removed members ---")
for i, j in removed_edges:
    print(f"({i}, {j})")

print("\n--- Compliance-based optimization history ---")
for iteration, state in enumerate(history):
    print(f"Iteration {iteration}: mass = {state['mass']:.4f} kg, "
          f"compliance = {state['compliance']:.4f} J, "
          f"members = {len(state['elements'])}")

# 10. Force-ranked greedy optimization (same constraint and baseline)
force_elements, force_history = force_based_optimize(
    nodes=nodes, elements=elements, E=E, A=A, rho=rho,
    loads=loads, fixed_dofs=fixed_dofs,
    max_compliance_increase=max_compliance_increase,
)

print("\n--- Algorithm comparison ---")
for name, states in [
    ("Compliance-based", history),
    ("Force-based", force_history),
]:
    final = states[-1]
    print(f"\n{name}")
    print(f"Final mass: {final['mass']:.6f} kg")
    print(f"Final compliance: {final['compliance']:.6f} J")
    print(f"Final members: {len(final['elements'])}")
    print(f"Mass reduction: {100 * (1 - final['mass'] / mass):.2f}%")
    print(f"Compliance increase: {100 * (final['compliance'] / compliance - 1):+.2f}%")
    print(f"Compliance limit satisfied: {final['compliance'] <= compliance_limit + 1e-10}")

# Verify each returned topology is the final recorded state
assert np.array_equal(optimized_elements, history[-1]['elements'])
assert np.array_equal(force_elements, force_history[-1]['elements'])
assert all(s['compliance'] <= compliance_limit + 1e-10 for s in history)
assert all(s['compliance'] <= compliance_limit + 1e-10 for s in force_history)

# 11. Plot baseline and both optimized topologies using identical axes
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True, sharey=True)
cases = [
    (f"Initial ground structure ({len(elements)} members)", elements),
    (f"Compliance-based ({len(optimized_elements)} members)", optimized_elements),
    (f"Force-based ({len(force_elements)} members)", force_elements),
]
for ax, (title, connectivity) in zip(axes, cases):
    for i, j in connectivity:
        ax.plot(nodes[[i, j], 0], nodes[[i, j], 1],
                color="tab:blue", linewidth=1.5)
    ax.scatter(nodes[:, 0], nodes[:, 1], color="black", s=16, zorder=3)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, L + 0.05)
    ax.set_ylim(-0.05, H + 0.05)
    ax.set_xlabel("x [m]")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("y [m]")
fig.tight_layout()
fig.savefig("topology_comparison.png", dpi=300)
plt.close(fig)

# 12. Save mass-compliance history for both algorithms
fig, ax = plt.subplots(figsize=(8, 5))
for label, states, marker in [
    ("Compliance-based", history, "o"),
    ("Force-based", force_history, "s"),
]:
    ax.plot(
        [state["mass"] for state in states],
        [state["compliance"] for state in states],
        marker=marker, markersize=3, label=label,
    )
ax.axhline(compliance_limit, color="tab:red", linestyle="--", label="5% compliance limit")
ax.set_xlabel("Structural mass [kg]")
ax.set_ylabel("Compliance [J]")
ax.set_title("Member-removal strategy comparison")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("algorithm_comparison.png", dpi=300)
plt.close(fig)

print("\nSaved: cantilever_deformation.png, topology_comparison.png, algorithm_comparison.png")

# 13. Stress and idealized member-level Euler buckling assessment
print("\n--- Stress and buckling assessment ---")

for name, connectivity in [
    ("Initial", elements),
    ("Compliance-based", optimized_elements),
    ("Force-based", force_elements),
]:
    check = assess_members(nodes, connectivity, E, A, loads, fixed_dofs)
    critical = int(np.argmin(check["buckling_factor"]))
    i, j = connectivity[critical]

    print(f"\n{name}")
    print(f"Maximum tensile stress: {check['maximum_tensile_stress'] / 1e6:.2f} MPa")
    print(f"Maximum compressive stress: {check['maximum_compressive_stress'] / 1e6:.2f} MPa")
    print(f"Minimum Euler buckling factor: {check['minimum_buckling_factor']:.3f}")
    print(f"Critical buckling member: ({i}, {j})")
    print(f"Critical member length: {check['lengths'][critical]:.4f} m")

print("\nNote: Euler factors assume straight, solid circular pin-ended members; ")
print("they do not establish global structural buckling capacity or material yield safety.")

# 14. Out-of-design load-case check: downward force at top-right node
alternative_loads = np.zeros_like(loads)
top_right_node = (ny - 1) * nx + (nx - 1)
alternative_loads[2 * top_right_node + 1] = -F

print("\n--- Alternative load case: top-right node ---")
alt_results = []
for name, connectivity in [
    ("Initial", elements),
    ("Compliance-based", optimized_elements),
    ("Force-based", force_elements),
]:
    sol = solve_truss(nodes, connectivity, E, A, alternative_loads, fixed_dofs)
    check = assess_members(nodes, connectivity, E, A, alternative_loads, fixed_dofs)
    displacement_mm = sol["u"][2 * top_right_node + 1] * 1e3
    alt_results.append((name, sol["compliance"]))
    print(f"{name}: tip displacement = {displacement_mm:.6f} mm, "
          f"compliance = {sol['compliance']:.6f} J, "
          f"maximum |stress| = {np.max(np.abs(check['stress'])) / 1e6:.2f} MPa, "
          f"minimum Euler factor = {check['minimum_buckling_factor']:.3f}")

baseline_alt = alt_results[0][1]
for name, value in alt_results[1:]:
    print(f"{name}: compliance change vs initial = "
          f"{100 * (value / baseline_alt - 1):+.2f}%")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([v[0] for v in alt_results], [v[1] for v in alt_results])
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_ylabel("Compliance [J]")
ax.set_title("Alternative load location: downward force at top-right node")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("alternative_load_comparison.png", dpi=300)
plt.close(fig)
print("Saved: alternative_load_comparison.png")
