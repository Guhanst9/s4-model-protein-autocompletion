import numpy as np
import torch
import torch.nn as nn

def transition(measure = 'legs', N=64, **kwargs):
    """
    computes HiPPO transition matricies

    Arguments:
        measure: 'legs' for scaled Legendre
        N: State dimension
    
    Returns: 
        A: [N, N] transition matrix
        B: [N, 1] input matrix
    """
    if (measure == 'legs'):
        return legs_matrix(N)
    else:
        return None # measure is not implemented
    
def legs_matrix(N):
    """
    HiPPO-LegS matrix for LRDs

    Uses scaled Legendre measure (optimal for learning LRD)
    """
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1

    # construct matrix, apply scaling, and return the HiPPO and Input matrices
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    
    # HiPPO matrix
    A = T @ M @ np.linalg.inv(T)
    
    # Input matrix
    B = np.sqrt(2 * q + 1)[:, None]
    
    return A, B

def nplr(measure = 'legs', N=64, **kwargs):
    """
    Normal-Plus-low-rank representation

    Returns matricies in NPLR form for efficiency
    """
    A, B = transition(measure=measure, N=N, **kwargs)

    # decompistion
    L, V = np.linalg.eig(A)
    L = L.astype(np.complex128)

    # low-rank correction
    P = np.conj(V)
    Q = V
    
    return L, P, Q, B

def hippo_init(nn_module, measure='legs', **kwargs):
    """
    Inits an S4 model with HiPPO

    Args:
        nn_module: S4 or S4D module to init
        measure: 'legs' for HiPPO-LegS
        **kwargs: Additional parameters
    """

    A, B = transition(measure=measure, N=nn_module.d_state, **kwargs)

    # initialize A using matrix
    with torch.no_grad():
        if hasattr(nn_module, 'A'):
            nn_module.A.data = torch.from_numpy(A).float()
        if hasattr(nn_module, 'A_log'):
            # for S4D, use log space
            eigenvals = np.real(np.linalg.eigvals(A))
            nn_module.A_log.data = torch.from_numpy(eigenvals).unsqueeze(0).repeat(nn_module.d_model, 1)
        if hasattr(nn_module, 'B'):
            nn_module.B.data = torch.from_numpy(B).T.float().repeat(nn_module.d_model, 1)

if __name__ == "__main__":
    # Test HiPPO initialization
    A, B = legs_matrix(N=16)
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"A eigenvalues: {np.real(np.linalg.eigvals(A))[:5]}")

