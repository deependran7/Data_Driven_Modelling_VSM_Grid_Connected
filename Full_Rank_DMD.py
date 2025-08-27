"""
opt_dmd_fullrank.py

Optimized Dynamic Mode Decomposition (optDMD) via Variable Projection
-- full-rank linear step (no SVD truncation) --
Author: adapted for user (based on Askham & Kutz varpro optDMD)

Dependencies:
    numpy, scipy

Usage:
    # Given X (n x m), t (m,), optional X_dot (n x m-1)
    alpha, Phi, b, recon, info = opt_dmd_fullrank(X, t, r=None, reg=1e-8, solver_opts=None, verbose=False)
"""

import numpy as np
from numpy.linalg import pinv, norm
from scipy.optimize import least_squares

# -------------------------
# Helpers
# -------------------------
def _build_exponential_matrix(t, alpha):
    """
    V[k,j] = exp(alpha_j * t_k)
    t: (m,)
    alpha: (r,) complex
    returns V: (m, r)
    """
    t = np.asarray(t).reshape(-1, 1)         # (m,1)
    alpha = np.asarray(alpha).reshape(1, -1) # (1,r)
    return np.exp(t @ alpha)                 # (m,r)

def _variable_projection_fullrank_solve(X, V, reg=0.0):
    """
    Full-rank linear solve for A in X ~= A V^H with optional Tikhonov regularization.

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
    # Use pseudoinverse for stability in case G is ill-conditioned
    Gpinv = pinv(G)
    A = X @ V @ Gpinv                     # (n, r)
    Xhat = A @ VH                         # (n, m)
    R = X - Xhat
    return A, R

def _pack_alpha(alpha):
    """complex alpha -> real vector (2r,)"""
    alpha = np.asarray(alpha)
    return np.concatenate([alpha.real, alpha.imag])

def _unpack_alpha(theta):
    """real vector -> complex alpha (r,)"""
    theta = np.asarray(theta)
    r = theta.size // 2
    return theta[:r] + 1j * theta[r:]

def _std_dmd_init_alpha_fullrank(X, t, r):
    """
    Use standard DMD (shifted snapshots) to build an initial alpha vector.
    This routine tries to provide r initial values; if X has fewer dynamic eigenvalues,
    the remaining entries are padded with zeros.
    NOTE: This uses shifted data X[:, :-1], X[:, 1:] and returns up to r eigenvalues.
    """
    n, m = X.shape
    if m < 2:
        return np.zeros(r, dtype=complex)
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    # if X1 is too small rank-wise, SVD will still work; we use full SVD but take min(r, rank)
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r0 = min(r, U.shape[1])
    Ur = U[:, :r0]
    Sr = S[:r0]
    Vr = Vh.conj().T[:, :r0]
    # low-rank Atilde
    Atilde = Ur.conj().T @ X2 @ Vr @ np.diag(1.0 / Sr)
    eigvals, _ = np.linalg.eig(Atilde)
    dt_med = np.median(np.diff(np.asarray(t)))
    # avoid zeros
    eigvals_safe = np.where(np.abs(eigvals) < 1e-16, 1e-16, eigvals)
    alpha0 = np.log(eigvals_safe) / dt_med
    if r0 < r:
        pad = np.zeros(r - r0, dtype=complex)
        alpha0 = np.concatenate([alpha0, pad])
    else:
        alpha0 = alpha0[:r]
    return alpha0

# -------------------------
# Main full-rank optDMD
# -------------------------
def opt_dmd_fullrank(X, t, r=None, alpha0=None, reg=0.0, solver_opts=None, verbose=False):
    """
    optDMD with full-rank linear step (no rank truncation).
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
        Tikhonov regularization parameter for linear solve (recommended for full-rank).
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
    info : dict (optimizer info and residual norm)
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
        alpha0 = _std_dmd_init_alpha_fullrank(X, t, r)

    theta0 = _pack_alpha(alpha0)

    def residual_theta(theta):
        alpha = _unpack_alpha(theta)           # (r,)
        V = _build_exponential_matrix(t, alpha) # (m, r)
        A, R = _variable_projection_fullrank_solve(X, V, reg=reg)
        # return concatenated real and imag residuals
        Rvec = R.ravel()
        return np.concatenate([Rvec.real, Rvec.imag])

    if solver_opts is None:
        solver_opts = {"ftol": 1e-9, "xtol": 1e-9, "gtol": 1e-9, "max_nfev": 2000}

    ls_res = least_squares(
        residual_theta, theta0, method="trf", jac='2-point',
        verbose=2 if verbose else 0, **solver_opts
    )

    alpha_opt = _unpack_alpha(ls_res.x)        # (r,)
    V_opt = _build_exponential_matrix(t, alpha_opt)
    A_opt, R_opt = _variable_projection_fullrank_solve(X, V_opt, reg=reg)

    # Split A_opt into mode shapes Phi (unit-norm columns) and amplitudes b
    b = np.array([norm(A_opt[:, j]) for j in range(A_opt.shape[1])], dtype=np.complex128)
    # avoid zero division
    b = np.where(b == 0, 1.0 + 0j, b)
    Phi = A_opt / b

    def reconstructor(t_new):
        Vnew = _build_exponential_matrix(np.asarray(t_new).reshape(-1), alpha_opt)  # (m_new, r)
        return Phi @ (np.diag(b) @ Vnew.conj().T)  # (n, m_new)

    info = {
        "cost": ls_res.cost,
        "nfev": ls_res.nfev,
        "status": ls_res.status,
        "message": ls_res.message,
        "residual_norm": norm(R_opt, 'fro'),
        "r_used": r,
        "reg": reg
    }
    return alpha_opt, Phi, b, reconstructor, info

