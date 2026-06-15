import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import os
import gmsh
import gmshtools
from gmshtools import MshHs

gmsh.initialize()
script_dir = os.path.dirname(os.path.abspath(__file__))
msh_path = os.path.join(script_dir, "..", "meshes", "Rechteck_Test.msh")
gmsh.open(msh_path)
netz = MshHs(gmsh.model)
gmsh.finalize()

# FEM Config:
# PDE:             2D equation: -d/dx(α₁·dΦ/dx) - d/dy(α₂·dΦ/dy) + β·Φ = f
# Elements:        Linear triangular elements (P1), 3 nodes per element
# Basis functions: Linear Lagrange functions over the triangle (N1, N2, N3)
# Quadrature:      1-point centroid rule (triangles), 1-point midpoint rule (Robin edges)
# Method:          Galerkin FEM
# BC:              Dirichlet (strong enforcement via inner_nodes)

# Definitions of the functions and parameters for the PDE:
# -d/dx(α₁(x,y) dΦ/dx) - d/dy(α₂(x,y) dΦ/dy) + β(x,y) Φ = f(x,y)   in G
# Φ(x,y): Scalar field that is searched for
# α₁,α₂ : Material property in x- and y-direction (diffusion coefficient)
# β     : Reaction coefficient
# f     : Source term

def alpha1(x, y): return y * x + 1
def alpha2(x, y): return x + y + 1
def beta(x, y):   return 2 * x**2
def f(x, y):      return x + y**2

# Dirichlet boundary value: Φ|∂G = phi_dirichlet(x,y)
def phi_dirichlet(x, y): return x**2 + y

# p list contains the [x, y] coordinates of the nodes
p = netz.points[:, :2]  # only x and y, ignore z if present

# t list contains the connectivity of the elements (3 node indices per triangle)
t = netz.Triangle.elements

# indices of the Dirichlet boundary nodes
print("Attribute:", [a for a in dir(netz) if not a.startswith('_')])
dR = np.unique(np.concatenate([
    netz.unten.nodes,
    netz.rechts.nodes,
    netz.oben.nodes,
    netz.links.nodes,
]))

N = len(p)

print(netz.unten.nodes)

#######################################################################################################
# Assemble the global stiffness matrix K and the right-hand side D over all triangular elements
K = lil_matrix((N, N))
D = np.zeros(N)

for elem in t:
    g = elem                                  # global node numbers of the 3 local nodes
    x = p[g, 0]
    y = p[g, 1]

    # signed area times two (determinant of the coordinate transformation)
    A2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    A2 = abs(A2)                              # use positive value, independent of orientation

    # gradient coefficients of the linear basis functions: grad(Nᵢ) = [bᵢ, cᵢ] / A2
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

    # evaluate the parameters in the midpoint (centroid) of the element
    xM = x.mean()
    yM = y.mean()
    a1M = alpha1(xM, yM)
    a2M = alpha2(xM, yM)
    bM  = beta(xM, yM)
    fM  = f(xM, yM)

    # local 3x3 element matrix and 3x1 element vector
    K_local = np.zeros((3, 3))
    for k in range(3):
        for j in range(3):
            # α₁ and α₂ stiffness contribution
            stiff = (a1M * b[k] * b[j] + a2M * c[k] * c[j]) / (2 * A2)
            # β mass contribution: A2/12 on the diagonal, A2/24 off the diagonal
            mass = bM * A2 / 12 if k == j else bM * A2 / 24
            K_local[k, j] = stiff + mass

    # source term, one-point rule: fM * Area / 3  with Area = A2/2
    D_local = np.full(3, fM * A2 / 6)

    # scatter the local contributions into the global system
    for k in range(3):
        D[g[k]] += D_local[k]
        for j in range(3):
            K[g[k], g[j]] += K_local[k, j]

# Convert to CSR for efficient column slicing and arithmetic
K_csr = csr_matrix(K)

# Dirichlet boundary condition assembly
# Move the known boundary nodes to the right-hand side: D ← D - Φ_d · K[:,d]
PhiR = phi_dirichlet(p[dR, 0], p[dR, 1])
D = D - K_csr[:, dR].toarray() @ PhiR

inner_nodes = [i for i in range(N) if i not in dR]
K_reduced = K_csr[np.ix_(inner_nodes, inner_nodes)]
D_reduced = D[inner_nodes]

# Solve the reduced linear system for the inner nodes using a sparse solver
Phi_inner = spsolve(K_reduced, D_reduced)

# Construct the full solution vector
Phi_complete = np.zeros(N)
for i, node in enumerate(inner_nodes):   # inner nodes
    Phi_complete[node] = Phi_inner[i]
for d, phi_d in zip(dR, PhiR):           # Dirichlet nodes
    Phi_complete[d] = phi_d

# Output the results
print("node  | x       | y       | Phi(x,y)")
print("-" * 45)
for i in range(N):
    print(f"{i:4d}  | {p[i,0]:7.3f} | {p[i,1]:7.3f} | {Phi_complete[i]:9.5f}")


# Plot the solution: filled contour for Φ, mesh overlay via gmshtools
netz.dim = 2
fig, ax = plt.subplots(figsize=(7, 5))

# 1) solution field as filled contour (no gmshtools equivalent for scalar fields)
tcf = ax.tricontourf(p[:, 0], p[:, 1], t, Phi_complete, levels=20, cmap='jet')
fig.colorbar(tcf, ax=ax, label='Φ(x,y)')

# 2) mesh overlay using gmshtools' own plot method
netz.Triangle.plot(ax=ax, color='k', alpha=0.15)

# 3) highlight the Dirichlet boundaries using gmshtools
for rand, col in [('unten', 'red'), ('rechts', 'blue'),
                  ('oben', 'green'), ('links', 'orange')]:
    getattr(netz, rand).plot(ax=ax, color=col)

ax.set_title('FEM 2D Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.show()
