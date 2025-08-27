"""
opt_dmd_fullrank.py

Optimized Dynamic Mode Decomposition (optDMD): From Paper: Askham, T., & Kutz, J. N. (2018). Variable projection methods for an optimized dynamic mode decomposition. SIAM Journal on Applied Dynamical Systems, 17(1), 380-416.
For the accuracy full rank has been used. 
Dependencies:
    numpy, scipy

Usage:
    # Given X (n x m), t (m,), optional X_dot (n x m-1)
    alpha, Phi, b, recon = opt_dmd_fullrank(X, t, r=None, reg=1e-8, solver_opts=None, verbose=False)
"""

import numpy as np
from numpy.linalg import pinv, norm
from scipy.optimize import least_squares


def build_exponential_matrix(t, alpha):
    """
    V[k,j] = exp(alpha_j * t_k)
    t: (m,)
    alpha: (r,) complex
    returns V: (m, r)
    """
    t = np.asarray(t).reshape(-1, 1)         # (m,1)
    alpha = np.asarray(alpha).reshape(1, -1) # (1,r)
    return np.exp(t @ alpha)                 # (m,r)

def variable_projection_fullrank_solve(X, V, reg=0.0):
    """
    Full-rank linear solve for A in X ~= A V^H 

    Minimizes || X - A V^H ||_F^2 + reg * ||A||_F^2
    Algebraic solution:
      A = X V (V^H V + reg I)^{-1}

    X: (n, m)
    V: (m, r)
    returns: A (n, r), R (n, m)
    """
    VH = V.conj().T                       # (r, m)
    G = VH @ V                            # (r, r)
    if reg is not None and reg > 0:
        G = G + reg * np.eye(G.shape[0])
    Gpinv = pinv(G)
    A = X @ V @ Gpinv                     # (n, r)
    Xhat = A @ VH                         # (n, m)
    R = X - Xhat
    return A, R

def pack_alpha(alpha):
    #complex alpha -> real vector (2r,)
    alpha = np.asarray(alpha)
    return np.concatenate([alpha.real, alpha.imag])

def unpack_alpha(theta):
    #real vector -> complex alpha (r,)
    theta = np.asarray(theta)
    r = theta.size // 2
    return theta[:r] + 1j * theta[r:]

def std_dmd_init_alpha_fullrank(X, t, r):
    #Use standard DMD to build an initial alpha vector.
    n, m = X.shape
    if m < 2:
        return np.zeros(r, dtype=complex)
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r0 = min(r, U.shape[1])
    Ur = U[:, :r0]
    Sr = S[:r0]
    Vr = Vh.conj().T[:, :r0]
  
    Atilde = Ur.conj().T @ X2 @ Vr @ np.diag(1.0 / Sr)
    eigvals, _ = np.linalg.eig(Atilde)
    dt_med = np.median(np.diff(np.asarray(t)))

    eigvals_safe = np.where(np.abs(eigvals) < 1e-16, 1e-16, eigvals)
    alpha0 = np.log(eigvals_safe) / dt_med
    if r0 < r:
        pad = np.zeros(r - r0, dtype=complex)
        alpha0 = np.concatenate([alpha0, pad])
    else:
        alpha0 = alpha0[:r]
    return alpha0


# Main full-rank optDMD

def opt_dmd_fullrank(X, t, r=None, alpha0=None, reg=0.0, solver_opts=None, verbose=False):
    """
    optDMD with full-rank linear step .
    Parameters
    ----------
    X : ndarray (n, m)
        Snapshot matrix (columns are snapshots at times t).
    t : ndarray (m,)
        Time stamps for columns of X.
    r : int or None
        Number of exponentials / modes to fit. If None -> full-rank r = min(n, m).
        WARNING: setting r = min(n,m) may be large; use reg>0 to stabilize.
    alpha0 : ndarray (r,), optional
        Initial continuous-time eigenvalues. If None, they are initialized using std DMD.
    reg : float
        Tikhonov regularization parameter for linear solve .
    solver_opts : dict, optional
        Passed to scipy.optimize.least_squares (e.g., max_nfev, ftol).
    verbose : bool
        Print optimizer progress.
    Returns
    -------
    alpha_opt : ndarray (r,) complex
    Phi : ndarray (n, r) complex (modes, columns normalized)
    b : ndarray (r,) complex (amplitudes)
    reconstructor : callable(t_new) -> X_rec (n, m_new)
    """
    X = np.asarray(X)
    t = np.asarray(t).reshape(-1)
    n, m = X.shape

    if r is None:
        r = min(n, m)
    else:
        # ensure r not larger than min(n,m)
        r = min(r, min(n, m))

    if alpha0 is None:
        alpha0 = std_dmd_init_alpha_fullrank(X, t, r)

    theta0 = pack_alpha(alpha0)

    def residual_theta(theta):
        alpha = unpack_alpha(theta)           # (r,)
        V = build_exponential_matrix(t, alpha) # (m, r)
        A, R = variable_projection_fullrank_solve(X, V, reg=reg)
        # return concatenated real and imag residuals
        Rvec = R.ravel()
        return np.concatenate([Rvec.real, Rvec.imag])

    if solver_opts is None:
        solver_opts = {"ftol": 1e-9, "xtol": 1e-9, "gtol": 1e-9, "max_nfev": 2000}
    #solving the least square problem for V(alpha)
    ls_res = least_squares(
        residual_theta, theta0, method="trf", jac='2-point', **solver_opts
    )

    alpha_opt = unpack_alpha(ls_res.x)        # (r,)
    V_opt = build_exponential_matrix(t, alpha_opt)
    A_opt, R_opt = variable_projection_fullrank_solve(X, V_opt, reg=reg)

    # Split A_opt into mode shapes Phi (unit-norm columns) and amplitudes b
    b = np.array([norm(A_opt[:, j]) for j in range(A_opt.shape[1])], dtype=np.complex128)
    # avoid zero division
    b = np.where(b == 0, 1.0 + 0j, b)
    Phi = A_opt / b

    def reconstructor(t_new):
        Vnew = build_exponential_matrix(np.asarray(t_new).reshape(-1), alpha_opt)  # (m_new, r)
        return Phi @ (np.diag(b) @ Vnew.conj().T)  # (n, m_new)

    return alpha_opt, Phi, b, reconstructor

