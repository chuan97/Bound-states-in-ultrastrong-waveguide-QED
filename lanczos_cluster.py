 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:46:33 2019

@author: juan
"""
import sys
import numpy as np
from numpy.linalg import eigh, norm
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

print('g:', float(sys.argv[1]), '\ndelta: ', float(sys.argv[2]), '\nN_exc: ', int(sys.argv[3]), '\nN_sites: ', int(sys.argv[4]), '\nlb_size: ', int(sys.argv[5]))

g = float(sys.argv[1])
delta = float(sys.argv[2])
N_exc = int(sys.argv[3])
N = int(sys.argv[4])
lb_size = int(sys.argv[5])

w0 = 1.
J = 0.4
ks = np.arange(N)
w = w0 - 2 * J * np.cos(2 * np.pi / N * ks)

def normalize(v):
    mod = norm(v)
    
    if mod == 0:
        return v, mod
    
    return v / mod, mod

def black_box_lanczos(H, lb_size):
    vals, vects = eigsh(H, lb_size)
    idx = np.argsort(vals)
    vals = vals[idx]
    vects, _ = normalize(vects[:,idx])
    
    return Eigensystem(vals, vects)

class Lanczos():
    def __init__(self, H, lb_size):
        self.H = H
        self.ob_size = H.shape[0]
        self.lb_size = lb_size

    def apply_H(self, v):
        v = self.H.dot(v) #the actual product

        return v
    
    def diagonalize(self): #original basis size and desired lanczos basis size
        lb_size = self.lb_size
        ob_size = self.ob_size
        
        phi = np.zeros((lb_size, ob_size), dtype = complex) #the lanczos basis, vectors stored as rows
        a = np.zeros(lb_size, dtype = complex) 
        n = np.zeros(lb_size, dtype = complex)

        phi[0] = np.random.rand(ob_size) #random initial vector for the lanczos basis
        phi[0], n[0] = normalize(phi[0])

        phi[1] = self.apply_H(phi[0]) # second vector in the lanczos basis
        a[0] = np.dot(np.conj(phi[0]), phi[1])
        phi[1] -= a[0] * phi[0]
        phi[1], n[1] = normalize(phi[1])
        
        #generating the next vector in the lanczos basis
        for i in range(1, lb_size - 1): 
            phi[i + 1] = self.apply_H(phi[i])
            a[i] = np.dot(np.conj(phi[i]), phi[i + 1])
            phi[i + 1] -= a[i] * phi[i] + n[i] * phi[i - 1]
            phi[i + 1], n[i + 1] = normalize(phi[i + 1])
            
            #reorthogonalization routine
            for j in range(i - 1): 
                q = np.dot(np.conj(phi[j]), phi[i + 1])
                phi[i + 1] -= q * phi[j]
            phi[i + 1], _ = normalize(phi[i + 1])      
            
        #self.phi = phi
        
        #constructing the lanczos matrix
        tridiag = np.diag(a) 
        tridiag += np.diag(n[1:], k = 1)
        tridiag += np.diag(n[1:], k = -1) 
        '''
        #alternatively making it sparse, requires sorted_eigensystem_sparse, excited state collapse extremely quickly
        tridiag = diags(a, 0, format = 'csr') 
        tridiag += diags(n[1:], 1, format = 'csr')
        tridiag += diags(n[1:], -1, format = 'csr') 
        '''
        #diagonalization of the lanzcos matrix
        eigsys_lb = sorted_eigsystem(tridiag) #eigenvectors expressed in the lanczos basis, as columns
        #self.eigsys_lb = eigsys_lb
        
        #change of basis of the eigenvector form the lanczos basis to the original basis
        vects_ob = np.dot(np.transpose(phi), eigsys_lb.vects)
        self.eigsys_ob = Eigensystem(eigsys_lb.vals, vects_ob) #eigenvectors expressed in the original basis, as columns

        return self.eigsys_ob

def sparse_kron_3(A, B, C):
    return sparse.kron(sparse.kron(A, B), C)

def sparse_kron_4(A, B, C, D):
    return sparse.kron(sparse.kron(A, B), sparse.kron(C, D))

def sparse_kron_6(A, B, C, D, E, F):
    return sparse.kron(sparse.kron(sparse.kron(A, B), sparse.kron(C, D)), sparse.kron(E, F))




def f_deltar(deltar, delta, g):
    return delta * np.exp(-2 * np.sum((-g / (np.sqrt(N) * (deltar + w))) ** 2))

def f_f(deltar, g):
    return -g / (np.sqrt(N) * (deltar + w))

def f_analisis(deltas, gs):
    deltar = 0

    analisis = {}

    for delta in deltas:
        delta = round(delta, 2)
        analisis[delta] = {}

        for g in gs: 
            g = round(g, 4)

            while True:
                change = abs(deltar - f_deltar(deltar, delta, g))
                deltar = f_deltar(deltar, delta, g)

                if change < 1e-9: #cuando la mejora es ya muy pequeña cierro el bucle
                    break 

            analisis[delta][g] = deltar #guardo los valores de deltar para poder usarlos más tarde

    return analisis

def sorted_eigsystem(H):
    vals, vects = eigh(H)
    
    return Eigensystem(vals, vects)

class Eigensystem:
    def __init__(self, vals, vects):
        self.vals = vals
        self.vects = vects
        self.size = len(vals)

def gen_a(dim):
    aux = np.zeros((dim - 1, dim - 1))
    
    for n in range(dim - 1):
        aux[n, n] = np.sqrt(n + 1)
    
    aux = np.append(aux, [np.zeros(dim - 1)], axis=0)
    aux = np.append(np.array([np.zeros(dim)]).T, aux, axis=1)
    
    return sparse.csr_matrix(aux)
  
    

class exact_Diag:
    def __init__(self, g, Delta=1, N_exc = 2):
        self.g = g
        self.Delta = Delta
        self.DeltaR = f_analisis([self.Delta], [self.g])[self.Delta][self.g]
        self.fk = f_f(self.DeltaR, self.g)
        self.L = self.fk.size
        self.N_exc = N_exc
        
        # Qubit operators
        sx = sparse.csr_matrix([[0,1],[1,0]], dtype = complex)
        sz = sparse.csr_matrix([[-1,0],[0,1]], dtype = complex)
        Hq = self.DeltaR * sparse.kron(sz, sparse.eye(self.N_exc ** self.L)) / 2.0
        # creation-anhilation
        a  = gen_a(self.N_exc)
        ad = a.T
        
        Hph = sparse.csr_matrix((2 * self.N_exc ** (self.L), 2 * self.N_exc ** (self.L)), dtype = complex)
        for i in range(self.L):
            Hph += w[i] * sparse_kron_4(sparse.eye(2), sparse.eye(self.N_exc ** i), np.dot(ad, a), sparse.eye(self.N_exc ** (self.L - i - 1)))
        
        Hc = sparse.csr_matrix((2 * self.N_exc ** (self.L), 2 * self.N_exc ** (self.L)), dtype = complex)
        for i in range(self.L):
            Hc += g / np.sqrt(self.L) * sparse_kron_4(sx, sparse.eye(self.N_exc ** i), a + ad, sparse.eye(self.N_exc ** (self.L - i - 1)))
        
        self.H = Hq + Hph + Hc
    
    def diag(self):
        lanczos = Lanczos(self.H, lb_size)
        self.eigsys = lanczos.diagonalize()
        #self.eigsys = black_box_lanczos(self.H, lb_size)
        
    def n_photons(self):
        try:
            print(self.eigsys.vals[2], self.eigsys.vals[1])
            a = gen_a(self.N_exc)
            ad = a.T

            self.GSphotons = np.zeros(self.L, dtype = 'complex')
            self.E1photons = np.zeros(self.L, dtype = 'complex')
            for n in range(self.L):
                for k in range(self.L):
                    for p in range(self.L):
                        if k < p:
                            Hkp = sparse_kron_6(sparse.eye(2), sparse.eye(self.N_exc ** k), ad, sparse.eye(self.N_exc ** (p - k - 1)), a, sparse.eye(self.N_exc ** (self.L - p - 1)))

                        elif k == p:
                            Hkp = sparse_kron_4(sparse.eye(2), sparse.eye(self.N_exc ** k), np.dot(ad, a), sparse.eye(self.N_exc ** (self.L - k - 1)))

                        elif k > p:
                            Hkp = sparse_kron_6(sparse.eye(2), sparse.eye(self.N_exc ** p), a, sparse.eye(self.N_exc ** (k - p - 1)), ad, sparse.eye(self.N_exc ** (self.L - k - 1)))
                            
                        self.GSphotons[n] += (1 / N) * np.exp(1j * 2 * np.pi * (k - p) * (n - self.L / 2) / self.L) * np.dot(np.conjugate(self.eigsys.vects[:, 2]).T, Hkp.dot(self.eigsys.vects[:, 2]))
                        self.E1photons[n] += (1 / N) * np.exp(1j * 2 * np.pi * (k - p) * (n - self.L / 2) / self.L) * np.dot(np.conjugate(self.eigsys.vects[:, 1]).T, Hkp.dot(self.eigsys.vects[:, 1]))
        
        except AttributeError:
            self.diag()
            self.n_photons()
            
    def save(self):
        filename = 'data_files/*lanczos_' + str(self.L) + '_' + str(self.N_exc) + '_' + str(self.Delta).replace('.', ',') + '_' + str(self.g).replace('.', ',') + '_' + str(lb_size) + '.txt'
        f = open(filename, 'w')
        
        for n in range(self.L):
            if n < self.L - 1:
                f.write(str(self.GSphotons[n].real) + ' ' + str(self.E1photons[n].real) + '\n')
            else:
                f.write(str(self.GSphotons[n].real) + ' ' + str(self.E1photons[n].real))
            
        f.close()

def kron_3(A, B, C):
    return sparse.kron(sparse.kron(A, B), C)

def kron_4(A, B, C, D):
    return sparse.kron(sparse.kron(A, B), sparse.kron(C, D))

def kron_6(A, B, C, D, E, F):
    return sparse.kron(sparse.kron(sparse.kron(A, B), sparse.kron(C, D)), sparse.kron(E, F))

def f_energy(deltar, g):
    f = f_f(deltar, g)
    return -0.5 * deltar + np.sum(f * (w * f + 2 * g / np.sqrt(N)))

H_exact = exact_Diag(g, delta, N_exc)
H_exact.n_photons()
H_exact.save()
