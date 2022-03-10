import math
import numpy as np

class sto3g :
     def __init__(self,zeta = 1.0):

         alpha_list = np.array([0.109818, 0.405771, 2.22766])
         self.d_list = np.array([0.444635, 0.535328, 0.154329])
         self.alpha_list = alpha_list * zeta ** 2

class atom:
    def __init__(self,center = np.array([0, 0, 0]),charge = 1, zeta = 1.0):
        self.center = center
        self.charge = charge
        self.sto3g = sto3g(zeta)

    @property
    def atomic_orbital(self):
        return self.center,self.sto3g

def F0(t : float) -> float:
    if t == 0:
        return 1
    else:
        return 0.5 * (math.pi / t) ** 0.5 * math.erf(t ** 0.5)

def overlap_integral(Psi_i, Psi_j) :

    (Ra, sto3g_a) = Psi_i
    (Rb, sto3g_b) = Psi_j
    overlap_integral = 0

    for i in range(3) :
        for j in range(3):

            alpha = sto3g_a.alpha_list[i]
            beta = sto3g_b.alpha_list[j]
            d1 = sto3g_a.d_list[i]
            d2 = sto3g_b.d_list[j]

            temp = d1 * d2 * \
                (4 * alpha * beta / math.pi ** 2) ** (3 / 4) * \
                (math.pi / (alpha + beta)) ** (3 / 2) * \
                math.exp(-alpha * beta / (alpha + beta) * math.dist(Ra,Rb) ** 2)
            
            overlap_integral += temp
    
    return overlap_integral

def overlap_matrix(atoms_list):
    overlap_matrix = np.mat(np.zeros((2,2)))
    for i in range(2):
        for j in range(2):
            Psi_i = atoms_list[i].atomic_orbital
            Psi_j = atoms_list[j].atomic_orbital

            overlap_matrix[i, j] = overlap_integral(Psi_i, Psi_j)

    return overlap_matrix

def kin_integral(Psi_i, Psi_j):

    (Ra, sto3g_a) = Psi_i
    (Rb, sto3g_b) = Psi_j
    kin_integral = 0

    for i in range(3) :
        for j in range(3):
            
            alpha = sto3g_a.alpha_list[i]
            beta = sto3g_b.alpha_list[j]
            d1 = sto3g_a.d_list[i]
            d2 = sto3g_b.d_list[j]

            temp = d1 * d2 * \
                (4 * alpha * beta / math.pi ** 2) ** (3 / 4) * \
                alpha * beta / (alpha + beta) * (3 - 2 * alpha * beta / (alpha + beta) * math.dist(Ra,Rb) ** 2) * \
                (math.pi / (alpha + beta)) ** (3 / 2) * \
                math.exp(-alpha * beta / (alpha + beta) * math.dist(Ra,Rb) ** 2)
            
            kin_integral += temp
    
    return kin_integral

def kin_matrix(atoms_list):
    kin_matrix = np.mat(np.zeros((2,2)))
    for i in range(2):
        for j in range(2):
            Psi_i = atoms_list[i].atomic_orbital
            Psi_j = atoms_list[j].atomic_orbital

            kin_matrix[i, j] = kin_integral(Psi_i, Psi_j)

    return kin_matrix

def nucatr_integral(Psi_i, Psi_j, Rc, Zc):

    (Ra, sto3g_a) = Psi_i
    (Rb, sto3g_b) = Psi_j
    nucatr_integral = 0
    for i in range(3) :
        for j in range(3):
            alpha = sto3g_a.alpha_list[i]
            beta = sto3g_b.alpha_list[j]
            d1 = sto3g_a.d_list[i]
            d2 = sto3g_b.d_list[j]
            Rp = (np.dot(alpha, Ra) + np.dot(beta, Rb)) / (alpha + beta)

            temp = d1 * d2 * \
                (4 * alpha * beta / math.pi ** 2) ** (3 / 4) * \
                -2 * math.pi / (alpha + beta) * Zc * \
                math.exp(-alpha * beta / (alpha + beta) * math.dist(Ra,Rb) ** 2) * \
                F0((alpha + beta) * math.dist(Rp,Rc) ** 2)

            nucatr_integral += temp
    return nucatr_integral

def nucatr_matrix(atoms_list, atom):
    nucatr_matrix = np.mat(np.zeros((2,2)))
    Zc = atom.charge
    Rc = atom.center
    for i in range(2):  
        for j in range(2):

            Psi_i = atoms_list[i].atomic_orbital
            Psi_j = atoms_list[j].atomic_orbital

            nucatr_matrix[i, j] = nucatr_integral(Psi_i, Psi_j, Rc, Zc)

    return nucatr_matrix

def nucatr_matrix_list(atoms_list):
    nucatr_matrix_list = [None] * 2
    for i in range(2):
        nucatr_matrix_list[i] = nucatr_matrix(atoms_list, atoms_list[i])
    
    return nucatr_matrix_list

def H_core_matrix(T, V_list):
    H_core = T.copy()
    for V in V_list:
        H_core += V

    return H_core

def two_elec_integral(Psi_i, Psi_j, Psi_k, Psi_l):

    Ra, sto3g_a = Psi_i
    Rb, sto3g_b = Psi_j
    Rc, sto3g_c = Psi_k
    Rd, sto3g_d = Psi_l

    two_elec_integral = 0

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    alpha = sto3g_a.alpha_list[i]
                    beta = sto3g_b.alpha_list[j]
                    gamma = sto3g_c.alpha_list[k]
                    delta = sto3g_d.alpha_list[l]
                    
                    d1 = sto3g_a.d_list[i]
                    d2 = sto3g_b.d_list[j]
                    d3 = sto3g_c.d_list[k]
                    d4 = sto3g_d.d_list[l]

                    Rp = (np.dot(alpha, Ra) + np.dot(beta, Rb)) / (alpha + beta)
                    Rq = (np.dot(gamma, Rc) + np.dot(delta, Rd)) / (gamma + delta)

                    temp = d1 * d2 * d3 * d4 * \
                        (16 * alpha * beta * gamma * delta / math.pi ** 4) ** (3 / 4) * \
                        2 * math.pi ** (5 / 2) / ((alpha + beta) * (gamma + delta) * (alpha + beta + gamma + delta) ** 0.5) * \
                        math.exp(-alpha * beta / (alpha + beta) * math.dist(Ra, Rb) ** 2 - gamma * delta / (gamma + delta) * math.dist(Rc, Rd) ** 2) * \
                        F0((alpha + beta) * (gamma + delta) / (alpha + beta + gamma + delta) * math.dist(Rp, Rq) ** 2)
                    
                    two_elec_integral += temp

    return two_elec_integral

def trans_matrix(S):

    eig, U = np.linalg.eig(S)
    X = np.mat(np.zeros((2,2)))
    for i in range(2):
        s_sqrt = math.sqrt(eig[i])
        X[:, i] = U[:, i] / s_sqrt
    return X

def density_matrix(C):

    c = C[:, 0]
    P = 2 * c * c.T

    return P

def G_matrix_element(P, atoms_list, i, j):
    g = 0
    for k in range(2):
        for l in range(2):
            miu = atoms_list[i].atomic_orbital
            nu = atoms_list[j].atomic_orbital
            lmbda = atoms_list[k].atomic_orbital
            sigma = atoms_list[l].atomic_orbital

            g += P[k, l] * (two_elec_integral(miu, nu, sigma, lmbda) - 0.5 * two_elec_integral(miu, lmbda, sigma, nu))
    return g

def G_matrix(P, atoms_list):
    G = np.mat(np.zeros((2,2)))
    for i in range(2):
        for j in range(2):
            G[i, j] = G_matrix_element(P, atoms_list, i, j)
    return G

def Fock_matrix(H_core, G):
    return H_core + G

def energy(P, H_core, F):
    E0 = 0
    for i in range(2):
        for j in range(2):
            E0 += 0.5 * P[j, i] * (H_core[i, j] + F[i, j])
    return E0

def Hartree_Fock(atoms_list,tol = 1e-6, max_cycle = 50):
    #init
    oldE0 = float('inf')
    
    S = overlap_matrix(atoms_list)
    T = kin_matrix(atoms_list)
    V_list = nucatr_matrix_list(atoms_list)
    H_core = H_core_matrix(T, V_list)

    X = trans_matrix(S)

    P = np.mat(np.zeros((2,2)))

    G = G_matrix(P, atoms_list)
    F = Fock_matrix(H_core, G)

    print("overlap matrix")
    print("S = ")
    print(S)
    print("kinetic energy matrix")
    print("T = ")
    print(T)
    print("nuclear attraction matrix")
    for i in range(2):
        print("V" + str(i) + " = ")
        print(V_list[i])
    print("core-Hamiltonian matrix")
    print("H_core = ")
    print(H_core)
    print("transformation matrix")
    print("X = ")
    print(X)
    print("two-electron part of the Fock matrix")
    print("G = ")
    print(G)
    print("Fock matrix (original basis set)")
    print("F = ")
    print(F)

    ###
    for i in range(max_cycle):
        #step 7
        new_F = X.T * F * X
        #step 8
        eig, new_C = np.linalg.eig(new_F)
        e = np.diag(eig)
        #step 9
        C = X * new_C
        #step 10
        P = density_matrix(C)
        #step
        G = G_matrix(P, atoms_list)
        F = Fock_matrix(H_core, G)

        E0 = energy(P, H_core, F)
        dE = E0 - oldE0
        oldE0 = E0

        #output
        print("\n")
        print("cycle " + str(i + 1))
        print("two-electron part of the Fock matrix")
        print("G = ")
        print(G)
        print("Fock matrix (original basis set)")
        print("F = ")
        print(F)
        print("Fock matrix (canonically orthonormalized basis set)")
        print("F' = ")
        print(new_F)
        print("coefficients matrix (canonically orthonormalized basis set)")
        print("C' = ")
        print(new_C)
        print("eigenvalues matrix")
        print("e = ")
        print(e)
        print("coefficients matrix (original basis set)")
        print("C = ")
        print(C)
        print("density matrix")
        print("P = ")
        print(P)
        print("E0 = ")
        print(E0) if abs(E0) < 10.0 else print("--")
        print("dE = ")
        print(dE) if abs(dE) < 10.0 else print("--")

        if abs(dE) < tol:
            break

He = atom([1.4632,0,0], 2, 2.0925)
H = atom([0,0,0], 1, 1.24)
atoms_list = [He, H]
Hartree_Fock(atoms_list)