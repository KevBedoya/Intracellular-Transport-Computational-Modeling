"""Mass series of the solver by exact powers of its one-step operator.

At v = 1000 the modal engine (computational_tools/mass_analysis_modal.py) cannot
be used at 96x96: the eigenvector matrix of the one-step operator has
cond(V) ~ 1e22 - 1e25 (fast advection along the tubes makes it nearly defective),
so the modal amplitudes are round-off and ln M comes out NaN for some pairs.

This module keeps everything else of that engine -- the same one-step map P,
built from the solver's own step functions and restricted to the same
symmetry-orbit basis (mam.build_reduced_operator) -- and replaces the
eigen-expansion with exact powers of P:

    x_k = P^k x_0,    M(k dT) = w . x_k.

P_s is entrywise nonnegative (checked on construction), so every product is a
sum of nonnegative terms: nothing cancels, and each entry of P^k keeps a relative
error of a few n eps however non-normal P is. This is the time-stepper's own
arithmetic (x_{k+1} = P x_k), regrouped:

  * ``samples``: the layer masses at k = j s, j = 0 .. J-1, from A = P^s, with
    blocks of B samples advanced together by A^B;
  * ``mass``: M at any real t, from binary powers P^(2^i) at k = floor(t / dT)
    and k + 1, ln M linear in between (as mam.ln_mass treats k as real);
  * ``rates``: lambda_1 and lambda_2 by the rule of launch._slowest_rates, from
    the eigenvalues of L_s = (P_s - I) / dT, with each mode's amplitude taken from
    its own left and right eigenvectors, a_n = (w . v_n)(u_n^H x_0) / (u_n^H v_n),
    instead of solving V c = x_0 -- plus each eigenvalue's condition number
    kappa_n = |u_n| |v_n| / |u_n^H v_n|.
"""
import numpy as np
import scipy.linalg as la

from computational_tools import mass_analysis_modal as mam

EXCITED_AMPLITUDE = 1e-9        # launch.CHAR_TIME_EXCITED_AMPLITUDE


def matrix_power(M, k):
    """M^k (k >= 1) by repeated squaring."""
    R, S = None, M
    while True:
        if k & 1:
            R = S.copy() if R is None else R @ S
        k >>= 1
        if not k:
            return R
        S = S @ S


class PowerEngine:
    """The one-step operator of one (a, b, v) point, on the symmetry-reduced state."""

    def __init__(self, G, a, b, v, N_LIST, domain_radius=1.0, D=1.0):
        layout = mam._Layout(G, G, np.sort(np.asarray(N_LIST, dtype=np.int64)))
        op = mam._Operator(layout, a, b, v, 0, domain_radius, D)
        x0 = mam.initial_state(layout)
        labels = mam.symmetry_orbits(layout, op.d_map)
        P, wD, wA, xs, resid = mam.build_reduced_operator(op, x0, labels)
        if resid > mam.INVARIANCE_TOL:
            raise RuntimeError(f"symmetry reduction not exact (residual {resid:.1e})")
        if P.min() < 0 or xs.min() < 0:
            raise RuntimeError("P_s or x_0 has negative entries; its powers would cancel")
        self.P, self.wD, self.wA, self.w, self.x0 = P, wD, wA, wD + wA, xs
        self.dT = op.dT
        self.n_state, self.n_reduced = layout.n, P.shape[0]

    def samples(self, s, J, block=256):
        """(J, 2) diffusive- and advective-layer masses at steps k = j s, j = 0 .. J-1."""
        A = matrix_power(self.P, s)
        W = np.vstack([self.wD, self.wA])
        X = np.empty((self.n_reduced, block))
        X[:, 0] = self.x0
        for i in range(1, block):
            X[:, i] = A @ X[:, i - 1]
        AB = matrix_power(A, block)
        out = np.empty((J, 2))
        j = 0
        while j < J:
            m = min(block, J - j)
            out[j:j + m] = (W @ X[:, :m]).T
            j += m
            if j < J:
                X = AB @ X
        return out

    def states(self, ks):
        """x at each integer step in ``ks``, as columns, by binary powers of P."""
        rem = np.asarray(ks, dtype=np.int64).copy()
        X = np.tile(self.x0[:, None], (1, rem.size))
        S = self.P
        while rem.any():
            sel = (rem & 1).astype(bool)
            if sel.any():
                X[:, sel] = S @ X[:, sel]
            rem >>= 1
            if rem.any():
                S = S @ S
        return X

    def layer_masses(self, ks):
        """(len(ks), 2) diffusive- and advective-layer masses at integer steps ``ks``."""
        return np.vstack([self.wD, self.wA]).dot(self.states(ks)).T

    def mass(self, t):
        """M(t) at real times, ln M linear between the steps floor(t / dT) and + 1."""
        k = np.asarray(t, dtype=np.float64) / self.dT
        k0 = np.floor(k).astype(np.int64)
        X = self.states(k0)
        M0, M1 = self.w @ X, self.w @ (self.P @ X)
        f = k - k0
        return np.exp((1 - f) * np.log(M0) + f * np.log(M1))

    def rates(self):
        """lambda_1, lambda_2 (Re, |Im|) as launch._slowest_rates defines them, with kappa_n."""
        n, dT = self.n_reduced, self.dT
        mu, vl, vr = la.eig((self.P - np.eye(n)) / dT, left=True, right=True)
        keep = 1 + dT * mu != 0
        mu, vl, vr = mu[keep], vl[:, keep], vr[:, keep]
        lam = -np.log1p(dT * mu.astype(complex)) / dT        # decay rate -log(beta) / dT
        uv = np.einsum("ij,ij->j", vl.conj(), vr)
        amp = (self.w @ vr) * (vl.conj().T @ self.x0) / uv
        kappa = np.linalg.norm(vl, axis=0) * np.linalg.norm(vr, axis=0) / np.abs(uv)
        order = np.argsort(lam.real)                          # slowest first
        lam, amp, kappa = lam[order], amp[order], kappa[order]
        lam1 = float(lam[0].real)
        excited = np.abs(amp) >= EXCITED_AMPLITUDE * np.abs(amp[0])
        others = excited & (np.abs(lam - lam[0]) > 1e-9 * abs(lam1))
        if not others.any():
            nan = float("nan")
            return dict(lambda_1=lam1, lambda_2=nan, lambda_2_imag=nan,
                        kappa_lambda_1=float(kappa[0]), kappa_lambda_2=nan)
        i2 = int(np.nonzero(others)[0][0])
        return dict(lambda_1=lam1, lambda_2=float(lam[i2].real),
                    lambda_2_imag=float(abs(lam[i2].imag)),
                    kappa_lambda_1=float(kappa[0]), kappa_lambda_2=float(kappa[i2]))
