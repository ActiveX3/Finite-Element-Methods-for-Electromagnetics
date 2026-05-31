import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# FEM Config:
# PDE:             1D Poisson equation: -d/dx(α·dφ/dx) + β·φ = f
# Elements:        Linear 1D elements (P1), 2 nodes per element
# Basis functions: Linear Lagrange hat functions (φ₁=(x₂-x)/Lₑ, φ₂=(x-x₁)/Lₑ)
# Quadrature:      1-point midpoint rule (evaluation at xM = (x1+x2)/2)
# Method:          Galerkin FEM
# BC:              Dirichlet (strong enforcement via inner_nodes) + Robin (weak form)

#p list contains the coordinates of the nodes
p=[1.75, 2.0, 1.25, 1.0, 1.5]

#t list contains the connectivity of the elements. Only works for linear elements
t = [[np.argsort(p)[i], np.argsort(p)[i+1]] for i in range(len(p)-1)]

# Definitions of the functions and parameters for the PDE:
# 1D Poisson-Equation: -d/dx(α(x) dφ/dx) + β(x) φ = f(x) in Ω
# φ(x): Scalar field: electric potential [V]
# α(x): Material property: Diffusion coefficient: Permittivity ε(x) [As/Vm]
# β(x): Material property: Reaction coefficient: 0 in electrostatics
# f(x): Source term: Volume charge density ρ(x) [C/m^3]
# q(x): boundary condition: Surface charge density σ(x) [C/m^2]
# γ(x): boundary condition: Robin coefficient: 0 for Neumann, ∞ for Dirichlet, finite for Robin [As/Vm]

def alpha(x): return x**2
def beta(x): return x
def f(x): return -x**3
def q(x): return 0
def gamma(x): return 0

# Boundary conditions
dR = [3, 1]
PhiR = [2.0, 6.0]
rR = []


#######################################################################################################
# calculate the local stiffness matrix for an element defined by its two nodes at coordinates x1 and x2
N = len(p)
K = lil_matrix((N, N))
D = np.zeros(N)

for i in range(len(t)):
    g1, g2 = t[i][0], t[i][1]
    x1, x2 = p[g1], p[g2]

    if x1 > x2:
        x1, x2 = x2, x1
        g1, g2 = g2, g1

    L_E = x2 - x1
    xM = (x1 + x2) / 2

    aM = alpha(xM)
    bM = beta(xM)
    fM = f(xM)

    K_diag = aM / L_E + bM * L_E / 3
    K_offdiag = -aM / L_E + bM * L_E / 6

    K_local = np.array([[K_diag, K_offdiag], [K_offdiag, K_diag]])
    D_local = np.array([fM * L_E / 2, fM * L_E / 2])

    K[g1,g1] += K_local[0,0]
    K[g1,g2] += K_local[0,1]
    K[g2,g1] += K_local[1,0]
    K[g2,g2] += K_local[1,1]

    D[g1] += D_local[0]
    D[g2] += D_local[1]

#Robin boundary condition Assembly
#     for each Robin-node r with αα·Φ' + γ·Φ = q follows:
#   - γ*φ is added to the diagonal of the stiffness matrix at the position of node r
#   - q is added to the right-hand side vector at the position of node r

for r in rR:
    x_r = p[r]
    K[r,r] += gamma(x_r)
    D[r] += q(x_r)

#Converts to CSR for efficient column slicing and arithmetic
K_csr = csr_matrix(K)

# Dirichlet boundary condition Assembly

for d, phi_d in zip(dR, PhiR):
    D = D -phi_d * K_csr[:,d]

inner_nodes = [i for i in range(N) if i not in dR]
K_reduced = K_csr[np.ix_(inner_nodes, inner_nodes)]
D_reduced = D[inner_nodes]

# Solve the linear system for the inner nodes using a sparse solver
Phi_inner = spsolve(K_reduced, D_reduced)

# Construct the full solution vector
Phi_complete = np.zeros(N)
for i, node in enumerate(inner_nodes):  #inner nodes
    Phi_complete[node] = Phi_inner[i]
for d, phi_d in zip(dR, PhiR):          #Dirichlet nodes
    Phi_complete[d] = phi_d

# Output the results
print("nodes  | x       | Phi(x)")
print("-" * 35)
for i in range(N):
    print(f"{i:5d} | {p[i]:7.3f} | {Phi_complete[i]:7.3f}")

p_array = np.array(p)
order = np.argsort(p_array)

plt.figure(figsize=(8, 5))
plt.plot(p_array[order], Phi_complete[order], marker='o', label='FEM Solution')
plt.title('FEM 1D Solution')
plt.ylabel('Φ(x)')
plt.grid(True)
plt.show()
