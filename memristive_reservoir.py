#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import random
import scipy.integrate as spint

def incidence(Adjacency_sparse):
    N = Adjacency_sparse.shape[0]
    M = Adjacency_sparse.nnz
    
    Incid = np.zeros((N,M))
    for node in range(N):
        beg_edge_idx, end_edge_idx = Adjacency_sparse.indptr[node], \
                                     Adjacency_sparse.indptr[node+1]
        connected_nodes = Adjacency_sparse.indices[beg_edge_idx:end_edge_idx]
        Incid[node,beg_edge_idx:end_edge_idx] = 1
        Incid[connected_nodes, range(beg_edge_idx,end_edge_idx)] = -1
    
    return Incid[:N-1, :]

def vertex_projector(Adjacency_sparse):
    
    Incid = incidence(Adjacency_sparse)
    BBT_inv = linalg.inv(Incid.dot(Incid.T))

    return np.einsum('ji,jk,kl', Incid, BBT_inv, Incid)


def cycle_projector(Adjacency_sparse):
    M = Adjacency_sparse.nnz
    return np.eye(M) - vertex_projector(Adjacency_sparse)


def CaravelliEqn(source_func, Omega_A, alpha, beta, chi):
    M = Omega_A.shape[0]
    def dwdt(t, w, above_mask, below_mask):
        Curr_proj = linalg.inv(np.eye(M) + chi*np.dot(Omega_A,np.diag(w)))
        deriv = -alpha*w - 1/beta*np.einsum('jk,k', Curr_proj, source_func(t))
        above_halt = np.logical_and(above_mask, deriv > 0)
        below_halt = np.logical_and(below_mask, deriv < 0)
        deriv[above_halt] = 0
        deriv[below_halt] = 0
        return deriv
    return dwdt


def create_adj_ER_F(n, m):
    edges = []
    num_edges = 0
    
    while num_edges < m:
        new_edge = random.sample(range(n), 2)
        new_edge.sort()
        if new_edge not in edges:
            edges.append(new_edge)
            num_edges += 1
    edges.sort()
    adjacency = sparse.lil_matrix((n, n), dtype='int')
    for i,j in edges:
        adjacency[i, j] = 1
    return adjacency.tocsr()

def create_adj_ER_G(n, p):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                edges.append([i, j])
            
    adjacency = sparse.lil_matrix((n, n), dtype='int')
    for i,j in edges:
        adjacency[i, j] = 1
    return adjacency.tocsr()

#==========================================================
# 2D Cubic
#==========================================================

def create_adj_cubic_2d(lattice_shape, undirected=True, xbias=1, ybias=1 ):
    """
    Returns an adjacency matrix for a 2D cubic lattice with number
    of nodes specified by lattice_shape.  If a directed network is
    requested with no bias, the default configuration is
    all bonds going from left to right and top to bottom. (recalling
    that we index nodes across rows then columns).  The xbias and
    ybias give the probability that a bond goes from left to
    right versus RL and top to bottom versus BT respectively.
    """
    num_ynodes, num_xnodes = lattice_shape
    num_nodes = num_xnodes * num_ynodes
    
    A = sparse.lil_matrix((num_nodes, num_nodes))
    
    # Form bond arrays to fill in row bonds and column bonds of the lattice
    x_bonds = np.ones(num_xnodes-1)
    y_bonds = np.ones(num_ynodes-1)
    
    # connect each row node to its neighbor to the right
    for first_row_node in range(0, num_nodes, num_xnodes):
         A[range(first_row_node, first_row_node + num_xnodes - 1),\
          range(first_row_node + 1, first_row_node + num_xnodes)] = x_bonds
    
    # connect each column node to its neighbor below
    for first_col_node in range(0, num_xnodes):
         A[range(first_col_node, num_nodes - num_xnodes, num_xnodes),\
          range(first_col_node + num_xnodes, num_nodes, num_xnodes)] = y_bonds
    
    # If we want an undirected network, just return the symmetrized form
    if undirected:
        A = A.tocsr()
        return A + A.T
    else:
        # If we want to toggle the direction of the elements (default direction is right and down)
        if (xbias != 1) or (ybias != 1):
            rows, cols = A.nonzero()
        
            for i, j in zip(rows, cols):
                if np.abs(i-j) == 1: # row bond
                    if np.random.rand() > xbias: # overcome the bias with probability 1-xbias
                        A[i, j] = 0
                        A[j, i] = 1
                else: #column bond
                    if np.random.rand() > ybias:
                        A[i, j] = 0
                        A[j, i] = 1
        return A.tocsr()

    
def integrate_Caravelli(dwdt, t_interval, w0):
    m = w0.shape[0]
    def w_above(t, w, above_mask):
        return np.max(w[np.logical_not(above_mask)]) - 1
    w_above.terminal = True

    def w_below(t, w, below_mask):
        return np.min(w[np.logical_not(below_mask)])
    w_below.terminal = True
    
    above_mask = np.zeros_like(w0)
    below_mask = np.zeros_like(w0)
    
    t, t_final = t_interval
    
    times = np.array([0])
    traj = w0.copy().reshape(m, 1)
    epsilon = 1e-8
    while t < t_final:
        dwdt_masked = lambda t, w: dwdt(t, w, above_mask, below_mask)
        w_above_masked = lambda t, w: w_above(t, w, above_mask)
        w_above_masked.terminal = True
        w_below_masked = lambda t, w: w_below(t, w, below_mask)
        w_below_masked.terminal = True
        sol = spint.solve_ivp(dwdt_masked, (t, t_final), w0, events=[w_above_masked, w_below_masked], max_step = 0.05)
        times = np.concatenate((times, sol.t[1:]))
        t = sol.t[-1]
        traj = np.hstack((traj, sol.y[:,1:]))
        w0 = sol.y[:,-1].copy()
        above_mask = (w0 >= 1-epsilon)
        w0[above_mask] = 1
        below_mask = (w0 <= epsilon)
        w0[below_mask] = 0
        
    return times, traj
