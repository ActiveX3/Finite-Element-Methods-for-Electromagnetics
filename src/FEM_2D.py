import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import os
import gmsh
from gmshtools import MshHs

basis = os.path.join(os.path.dirname(os.path.abspath(__file__)),"2D_Reference")
p     = np.loadtxt(os.path.join(basis, "Netz2D_p.dat"), dtype=float)
t     = np.loadtxt(os.path.join(basis, "Netz2D_t.dat"), dtype=int)
dR    = np.loadtxt(os.path.join(basis, "Netz2D_dr_b).dat"),dtype=int)
sol_a = np.loadtxt(os.path.join(basis, "Netz2D_LoesungB.dat"),dtype=float)
rrb   = np.loadtxt(os.path.join(basis, "Netz2D_rr_b).dat"),dtype=int)


print("p:", p.shape)
print("t:", t.shape, "min", t.min(), "max", t.max())
print("dR_a:", dR.shape, "min", dR.min(), "max", dR.max())
print("sol_a:", sol_a.shape, "| N =", p.shape[0])

# gmsh.initialize()
# mesh_dir = os.path.dirname(os.path.abspath(__file__))
# msh_path = os.path.join(script_dir, "..", "meshes", "Meshname.msh")
# gmsh.open(msh_path)
# netz = MshHs(gmsh.model)
# gmsh.finalize()

# FEM Config:
# PDE:             2D equation: -d/dx(α₁·dΦ/dx) - d/dy(α₂·dΦ/dy) + β·Φ = f
# Elements:        Linear triangular elements (P1), 3 nodes per element
# Basis functions: Linear Lagrange functions over the triangle (N1, N2, N3)
# Quadrature:      1-point centroid rule (triangles), 1-point midpoint rule (Robin edges)
# Method:          Galerkin FEM
# BC:              Dirichlet BC (on ∂G_D):                               Φ = Φ_D
#                  Neumann   BC (on ∂G_N):       α₁·dΦ/dx·nₓ + α₂·dΦ/dy·nᵧ = q         (equals Robin with γ = 0 )
#                  Robin     BC (on ∂G_R): α₁·dΦ/dx·nₓ + α₂·dΦ/dy·nᵧ + γ·Φ = q 
#                  

# Definitions of the functions and parameters for the PDE:
# -d/dx(α₁(x,y) dΦ/dx) - d/dy(α₂(x,y) dΦ/dy) + β(x,y) Φ = f(x,y)   in G
# Φ(x,y): Scalar field that is searched for
# α₁,α₂ : Material property in x- and y-direction (diffusion coefficient)
# β     : Reaction coefficient
# f     : Source term

def alpha1(x, y):
    if 1.25 <= x <= 1.75 and 3<=y<= 3.5:
        return 0.01
    else:
        return 5*y*x**2


def alpha2(x, y): return y**2

def beta(x, y):   return 500 if (x-1.5)**2 + (y-1.75)**2 <= 0.35**2 else 5 #npwhere todo with bigger matrices

def f(x, y): return -10*x*y if x>=2 else 0

# Dirichlet boundary value: Φ|∂G = phi_dirichlet(x,y)
def phi_dirichlet(x, y): return x**2 - y**2 + 1 

# Robin boundary:  alpha*dPhi/dn + gamma*Phi = q   on the selected boundary curves
def gamma_robin(x, y): return 3*x*y      # Robin coefficient

def q_robin(x, y):     return 20      # Robin source term 


# Boundary-curve configuration                            
# Each boundary curve is either Dirichlet OR Robin, never both.
# To make a curve Robin, list its name here; it is then removed from Dirichlet.
all_boundary_curves = []
robin_curves        = []          # e.g. ['rechts'] makes the right edge Robin
dirichlet_curves    = []

# Mesh daten
# p:  [x, y] coordinates of the nodes
# t:  connectivity of the triangular elements (3 node indices per triangle)

N = len(p)
print(t.min(), t.max(), p.shape[0])
#print("Attribute:", [a for a in dir(netz) if not a.startswith('_')])

# indices of the Dirichlet boundary nodes (only from the non-Robin curves)
# if dirichlet_curves:
#     dR = np.unique(np.concatenate([
#         getattr(netz, c).nodes for c in dirichlet_curves
#     ]))
# else:
#     dR = np.array([], dtype=int)'

#   Assemble the global stiffness matrix K and right-hand side D (triangles)
K = lil_matrix((N, N))
D = np.zeros(N)

for elem in t:
    g = elem                                  # global node numbers of the 3 local nodes
    x = p[g, 0]
    y = p[g, 1]

    # signed area times two (determinant of the coordinate transformation)
    A2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    A2 = abs(A2)                              # positive value, independent of orientation

    # gradient coefficients of the linear basis functions: grad(Ni) = [bi, ci] / A2
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

    # evaluate the parameters at the centroid of the element
    xM = x.mean()
    yM = y.mean()
    a1M = alpha1(xM, yM)
    a2M = alpha2(xM, yM)
    bM  = beta(xM, yM)
    fM  = f(xM, yM)

    # local 3x3 element matrix
    K_local = np.zeros((3, 3))
    for k in range(3):
        for j in range(3):
            # alpha1 / alpha2 stiffness contribution
            stiff = (a1M * b[k] * b[j] + a2M * c[k] * c[j]) / (2 * A2)
            # beta mass contribution: A2/12 on the diagonal, A2/24 off the diagonal
            mass = bM * A2 / 12 if k == j else bM * A2 / 24
            K_local[k, j] = stiff + mass

    # source term, one-point rule: fM * Area / 3  with Area = A2/2
    D_local = np.full(3, fM * A2 / 6)

    # scatter the local contributions into the global system
    for k in range(3):
        D[g[k]] += D_local[k]
        for j in range(3):
            K[g[k], g[j]] += K_local[k, j]


#         Robin boundary assembly     
# The Robin term is a line integral along the boundary curve:
#   matrix:  int gamma * N_k * N_j ds   (2x2 edge mass matrix: L/3 diag, L/6 off-diag)
#   rhs:     int q     * N_k       ds   (L/2 each)
# This is exactly a 1D FEM assembly along the Robin boundary.

for edge in rrb:
    n1, n2 = edge[0], edge[1]
    x1, y1 = p[n1]
    x2, y2 = p[n2]
    L = np.hypot(x2 - x1, y2 - y1)            # length of the boundary edge

    xM = (x1 + x2) / 2                        # edge midpoint, 1-point quadrature
    yM = (y1 + y2) / 2
    gM = gamma_robin(xM, yM)
    qM = q_robin(xM, yM)

    # gamma -> matrix (edge mass matrix)
    K[n1, n1] += gM * L / 3
    K[n2, n2] += gM * L / 3
    K[n1, n2] += gM * L / 6
    K[n2, n1] += gM * L / 6

    # q -> right-hand side
    D[n1] += qM * L / 2
    D[n2] += qM * L / 2

# Dirichlet boundary condition
# Convert to CSR
K_csr = csr_matrix(K)
K_ref = np.loadtxt(basis + r"\All2D_K_ohneRobin.sec")
D_ref = np.loadtxt(basis + r"\All2D_D_ohneRobin.sec")
print("K_ref:", K_ref.shape)
print("K-Differenz:", np.max(np.abs(K_csr.toarray() - K_ref)))
print("D-Differenz:", np.max(np.abs(D - D_ref)))



# Move the known boundary values to the right-hand side: D <- D - K[:,dR] @ PhiR
if len(dR) > 0:
    PhiR = phi_dirichlet(p[dR, 0], p[dR, 1])
    D = D - K_csr[:, dR].toarray() @ PhiR
else:
    PhiR = np.array([])

# inner (unknown) nodes = all nodes that are not Dirichlet
is_dirichlet = np.zeros(N, dtype=bool)
is_dirichlet[dR] = True
inner_nodes = np.where(~is_dirichlet)[0]

K_reduced = K_csr[np.ix_(inner_nodes, inner_nodes)]
D_reduced = D[inner_nodes]  



### Solve the reduced system  ###
#################################

Phi_inner = spsolve(K_reduced, D_reduced)

# Construct the full solution vector
Phi_complete = np.zeros(N)
Phi_complete[inner_nodes] = Phi_inner
for d, phi_d in zip(dR, PhiR):           # write back the Dirichlet values
    Phi_complete[d] = phi_d

fehler = np.max(np.abs(sol_a - Phi_complete))
print(f"max Fehler: {fehler:.2e}  (Ziel < 1e-12)")
mean_fehler = np.mean(np.abs(sol_a - Phi_complete))
print(f"mittlerer Fehler: {mean_fehler:.2e}")

###   PLOT  ###
###############
fig, ax = plt.subplots(figsize=(7, 5))

# Error plot
fig_err, ax_err = plt.subplots()
ax_err.plot(sol_a - Phi_complete)
ax_err.set_xlabel("Knoten")
ax_err.set_ylabel("Fehler")
ax_err.set_title("Fehler: sol_b - Phi_complete")


# FEM plot
ax.plot(p[np.unique(rrb), 0], p[np.unique(rrb), 1], 'b.', ms=4)                          #highlighting the robin nodes
tcf = ax.tricontourf(p[:, 0], p[:, 1], t, Phi_complete, levels=40, cmap='jet')           # highlights the color gradients
ax.triplot(p[:, 0], p[:, 1], t, color='k', alpha=0.15)                                   
fig.colorbar(tcf, ax=ax, label='Phi(x,y)')                                               #
ax.tricontour(p[:, 0], p[:, 1], t, Phi_complete, levels=100, colors='k', linewidths=0.5)  # highlights the lines 
ax.tricontour(p[:, 0], p[:, 1], t, Phi_complete, levels=[0], colors='r', linewidths=0.5) # hightlights the 0 potential

ax.set_title('FEM 2D Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal', adjustable="box")

plt.show()