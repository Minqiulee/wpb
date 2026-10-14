"""Weighted Poisson-Binomial (WPB) distribution.

Models the distribution of the weighted sum

    S = sum_i w_i * X_i,   X_i ~ Bernoulli(p_i)  (independent trials)

and evaluates the percentile of an observed outcome on its CDF.

Two computation paths are provided:

1. **Exact (FFT / characteristic function)** — builds the probability
   mass function over the full support ``{0, ..., sum(w)}`` by
   multiplying per-trial characteristic functions on an FFT grid and
   inverting with an inverse FFT.  Trials sharing the same ``(p, w)``
   pair are grouped and raised to their multiplicity, so the grid-build
   step costs ``O(G * N)`` with ``G`` the number of distinct ``(p, w)``
   pairs (<= 32 * n_distinct_weights in the intended use case) instead
   of the ``O(n * S)`` of the classical dynamic-programming recursion;
   the transform itself is ``O(N log N)`` with
   ``N = next_fast_len(S + 1)``.

2. **Saddlepoint (Lugannani-Rice)** — approximates ``P(S <= s)`` at a
   single point directly from the cumulant generating function
   ``K(t) = sum_i log(1 - p_i + p_i * e^(t * w_i))``.  Memory is
   ``O(n)`` (no PMF array is ever materialised), which is the point of
   this path: when weights are large the exact PMF needs
   ``O(sum(w))`` floats, while the saddlepoint only touches the raw
   probability/weight arrays.  Trade-off: it is an *approximation* —
   relative error is typically O(1/n) away from the mean but grows in
   the far tails and for small n; see :func:`saddlepoint_cdf`.

Numerical precision notes
-------------------------
* The FFT path is exact up to floating-point round-off.  For large
  supports the inverse FFT can leave tiny negative masses (~1e-15);
  they are clipped to zero and the PMF is renormalised.
* Characteristic-function values are computed in polar form per group
  (magnitude ``r**m``, angle ``m * atan2(...)``) which is numerically
  preferable to naive complex powering and avoids log-branch issues.
* The saddlepoint path solves ``K'(t) = s`` with a safeguarded
  Newton-bisection hybrid in float64; ``K(t)`` and its derivatives are
  evaluated in the numerically stable ``log1p``/logistic form.

Edge-case policy
----------------
* ``p_i == 0`` or ``p_i == 1``: deterministic trials are peeled off and
  contribute a fixed shift ``sum(w_i * p_i)`` to the support; they do
  not participate in the convolution or the CGF solve.
* ``w_i == 0``: the trial cannot change the weighted sum regardless of
  its outcome, so it is dropped (mathematically exact, not an
  approximation).  A ``UserWarning`` is emitted.
* Empty input: the sum is degenerate at 0, PMF is ``[1.0]`` and the
  CDF/percentile of an empty observation is 1.0.
* Mismatched lengths, non-finite probabilities, probabilities outside
  ``[0, 1]``, negative or non-integer weights: ``ValueError``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numba import njit, prange

__all__ = [
    "validate_inputs",
    "wpb_pmf",
    "wpb_cdf",
    "saddlepoint_cdf",
    "percentile_of_outcome",
]


# ---------------------------------------------------------------------------
# Validation / preprocessing
# ---------------------------------------------------------------------------

def validate_inputs(
    p: np.ndarray,
    w: np.ndarray,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and normalise WPB inputs.

    Parameters
    ----------
    p : (n,) array_like of float
        Per-trial success probabilities.  Must be finite and in
        ``[0, 1]``.
    w : (n,) array_like of int
        Per-trial weights.  Must be non-negative integers (a float
        array is accepted if every entry is exactly integral).
        Zero weights are allowed; the corresponding trials are dropped
        downstream because they cannot affect the weighted sum.
    x : (n,) array_like of bool, optional
        Observed per-trial outcomes.  If given, must have the same
        length as ``p`` and contain only 0/1 (or boolean) values.

    Returns
    -------
    p, w, x
        ``p`` as float64, ``w`` as int64, ``x`` as bool (or ``None``).

    Raises
    ------
    ValueError
        On length mismatch, non-finite or out-of-range probabilities,
        negative/non-integral weights, or non-binary observations.
    """
    p = np.asarray(p, dtype=np.float64)
    w_raw = np.asarray(w)

    if p.ndim != 1 or w_raw.ndim != 1:
        raise ValueError("p and w must be 1-D arrays")
    if p.shape[0] != w_raw.shape[0]:
        raise ValueError(
            f"length mismatch: p has {p.shape[0]} entries, "
            f"w has {w_raw.shape[0]}"
        )
    if not np.all(np.isfinite(p)):
        raise ValueError("p contains non-finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p entries must lie in [0, 1]")

    if not np.issubdtype(w_raw.dtype, np.integer):
        w_f = np.asarray(w_raw, dtype=np.float64)
        if not np.all(np.isfinite(w_f)) or not np.all(w_f == np.floor(w_f)):
            raise ValueError("weights must be integers")
        w_raw = w_f
    w = w_raw.astype(np.int64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    if x is None:
        return p, w, None

    x = np.asarray(x)
    if x.shape != p.shape:
        raise ValueError(
            f"length mismatch: x has shape {x.shape}, expected {p.shape}"
        )
    if x.dtype != bool:
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("x must be boolean (or 0/1 valued)")
        x = x.astype(bool)
    return p, w, x


def _strip_degenerate(
    p: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove zero-weight and deterministic trials.

    Returns the stochastic sub-problem ``(p_s, w_s)`` plus the fixed
    shift ``sum(w_i * p_i)`` contributed by deterministic trials.  Both
    removals are mathematically exact.
    """
    nonzero = w > 0
    if not np.all(nonzero):
        n_dropped = int(np.count_nonzero(~nonzero))
        warnings.warn(
            f"{n_dropped} trial(s) with w_i == 0 dropped "
            "(they cannot affect the weighted sum)",
            UserWarning,
            stacklevel=2,
        )
        p, w = p[nonzero], w[nonzero]

    stochastic = (p > 0.0) & (p < 1.0)
    shift = int(np.dot(w[~stochastic], p[~stochastic].astype(np.int64))) \
        if np.any(~stochastic) else 0
    return p[stochastic], w[stochastic], shift


# ---------------------------------------------------------------------------
# Numba kernels: characteristic function on the FFT grid
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _cf_grid(
    gp: np.ndarray, gw: np.ndarray, gm: np.ndarray, n_grid: int
) -> np.ndarray:
    """Product of per-group characteristic functions on the unit circle.

    For each distinct ``(p, w)`` group with multiplicity ``m`` the
    factor ``(1 - p + p * z**w)**m`` is evaluated at the ``n_grid``
    roots of unity ``z = exp(2*pi*i*k/n_grid)`` in polar form and
    multiplied into the output.  Runs in ``O(G * n_grid)`` with ``G``
    the number of groups.
    """
    out = np.ones(n_grid, dtype=np.complex128)
    two_pi_over_n = 2.0 * math.pi / n_grid
    for g in range(gp.shape[0]):
        pj = gp[g]
        qj = 1.0 - pj
        wj = gw[g]
        m = gm[g]
        for k in prange(n_grid):
            # Negative sign: with this convention the grid holds the
            # forward DFT of the PMF, so np.fft.ifft inverts it.
            # (k * wj) mod n_grid keeps the trig argument small.
            theta = -two_pi_over_n * ((k * wj) % n_grid)
            re = qj + pj * math.cos(theta)
            im = pj * math.sin(theta)
            r = math.sqrt(re * re + im * im)
            ang = math.atan2(im, re)
            rm = r ** m
            angm = ang * m
            out[k] *= complex(rm * math.cos(angm), rm * math.sin(angm))
    return out


def _next_fast_len(n: int) -> int:
    """Smallest 5-smooth length >= n (pocketFFT-friendly)."""
    if n <= 1:
        return 1
    candidate = n
    while True:
        m = candidate
        for prime in (2, 3, 5):
            while m % prime == 0:
                m //= prime
        if m == 1:
            return candidate
        candidate += 1


def _group_trials(
    p: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group trials by distinct (p, w) pair; return (p_g, w_g, mult_g)."""
    keys = np.lexsort((w, p))
    p_s, w_s = p[keys], w[keys]
    is_new = np.ones(p.shape[0], dtype=bool)
    is_new[1:] = (p_s[1:] != p_s[:-1]) | (w_s[1:] != w_s[:-1])
    starts = np.flatnonzero(is_new)
    mult = np.diff(np.append(starts, p.shape[0])).astype(np.int64)
    return p_s[starts], w_s[starts], mult


# ---------------------------------------------------------------------------
# Exact PMF / CDF via FFT
# ---------------------------------------------------------------------------

def wpb_pmf(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exact PMF of the weighted sum over its full support.

    Parameters
    ----------
    p : (n,) array_like of float
        Per-trial success probabilities in ``[0, 1]``.
    w : (n,) array_like of int
        Positive integer per-trial weights (zeros are dropped).

    Returns
    -------
    pmf : (S + 1,) ndarray of float64
        ``pmf[s] = P(S = s)`` for ``s = 0, ..., S`` where
        ``S = sum(w)``.  The array is indexed directly by the weighted
        outcome value.

    Notes
    -----
    Complexity is ``O(G * N + N log N)`` where ``G`` is the number of
    distinct ``(p, w)`` pairs and ``N`` the padded FFT length
    (``~sum(w)``).  Memory is ``O(N)``.  For huge supports consider
    :func:`saddlepoint_cdf`, which needs only ``O(n)`` memory.
    """
    p, w, _ = validate_inputs(p, w)
    p_s, w_s, shift = _strip_degenerate(p, w)

    if p_s.size == 0:
        pmf = np.zeros(shift + 1, dtype=np.float64)
        pmf[shift] = 1.0
        return pmf

    support = int(w_s.sum())  # max sum of the stochastic part
    n_grid = _next_fast_len(support + 1)
    gp, gw, gm = _group_trials(p_s, w_s)

    cf = _cf_grid(gp, gw, gm, n_grid)
    pmf_stoch = np.fft.ifft(cf).real[: support + 1]

    # Clean up FFT round-off: clip tiny negatives, renormalise.
    pmf_stoch[pmf_stoch < 0.0] = 0.0
    total = pmf_stoch.sum()
    if total > 0.0:
        pmf_stoch /= total

    # Index equals the weighted outcome value: pad to the full 0..sum(w)
    # range so trailing zeros (unachievable high sums) are preserved.
    full_support = int(w.sum())
    pmf = np.zeros(full_support + 1, dtype=np.float64)
    pmf[shift : shift + support + 1] = pmf_stoch
    return pmf


def wpb_cdf(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exact CDF of the weighted sum over its full support.

    Parameters
    ----------
    p, w
        See :func:`wpb_pmf`.

    Returns
    -------
    cdf : (S + 1,) ndarray of float64
        ``cdf[s] = P(S <= s)``; the index equals the weighted outcome
        value.  ``cdf[-1] == 1.0`` up to float64 round-off.
    """
    cdf = np.cumsum(wpb_pmf(p, w))
    # Guard against drift past 1.0 from cumsum round-off.
    cdf[cdf > 1.0] = 1.0
    return cdf


# ---------------------------------------------------------------------------
# Saddlepoint (Lugannani-Rice) CDF approximation
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _kgf(t: float, p: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Cumulant generating function K(t) and its first two derivatives.

    Uses the logistic parametrisation for numerical stability:
    ``K_i = softplus(logit(p_i) + t*w_i) - softplus(logit(p_i))``,
    ``K'_i = w_i * sigmoid(logit(p_i) + t*w_i)``,
    ``K''_i = w_i**2 * sigmoid'(logit(p_i) + t*w_i)``.
    """
    K = 0.0
    K1 = 0.0
    K2 = 0.0
    for i in range(p.shape[0]):
        pi = p[i]
        wi = w[i]
        logit = math.log(pi) - math.log1p(-pi)
        u = logit + t * wi
        # softplus(u), sigmoid(u) with overflow-safe branching
        if u > 0.0:
            sp = u + math.log1p(math.exp(-u))
            sig = 1.0 / (1.0 + math.exp(-u))
        else:
            sp = math.log1p(math.exp(u))
            eu = math.exp(u)
            sig = eu / (1.0 + eu)
        if logit > 0.0:
            sp0 = logit + math.log1p(math.exp(-logit))
        else:
            sp0 = math.log1p(math.exp(logit))
        K += sp - sp0
        K1 += wi * sig
        K2 += wi * wi * sig * (1.0 - sig)
    return K, K1, K2


@njit(cache=True)
def _solve_saddlepoint(
    s: float, p: np.ndarray, w: np.ndarray, tol: float = 1e-12, max_iter: int = 64
) -> float:
    """Solve K'(t) = s with a safeguarded Newton-bisection hybrid.

    K' is strictly increasing from 0 (t -> -inf) to sum(w) (t -> +inf),
    so a unique root exists for any interior s.  Returns 0.0 for s at
    the mean (where the Lugannani-Rice formula is used in its limiting
    form by the caller).
    """
    lo, hi = -1.0, 1.0
    while _kgf(lo, p, w)[1] > s:
        lo *= 2.0
    while _kgf(hi, p, w)[1] < s:
        hi *= 2.0

    t = 0.5 * (lo + hi)
    for _ in range(max_iter):
        _, K1, K2 = _kgf(t, p, w)
        f = K1 - s
        if abs(f) < tol:
            return t
        if f > 0.0:
            hi = t
        else:
            lo = t
        # Newton step, fall back to bisection if it leaves the bracket
        if K2 > 0.0:
            t_newton = t - f / K2
            if lo < t_newton < hi:
                t = t_newton
            else:
                t = 0.5 * (lo + hi)
        else:
            t = 0.5 * (lo + hi)
    return t


@njit(cache=True)
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@njit(cache=True)
def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def saddlepoint_cdf(p: np.ndarray, w: np.ndarray, s: float) -> float:
    """Lugannani-Rice approximation of ``P(S <= s)``.

    Parameters
    ----------
    p : (n,) array_like of float
        Per-trial success probabilities.
    w : (n,) array_like of int
        Positive integer weights.
    s : float
        Evaluation point (weighted sum value).  A continuity
        correction of 0.5 is applied internally because S lives on a
        lattice (span 1 when ``gcd(w) == 1``).

    Returns
    -------
    float
        Approximate ``P(S <= s)`` in ``[0, 1]``.

    Notes
    -----
    **Why this path exists:** memory.  The exact FFT CDF materialises
    ``sum(w) + 1`` floats; here only the raw ``(p, w)`` arrays are
    touched (``O(n)`` memory, ``O(n)`` per Newton iteration).

    **Accuracy assumptions / trade-offs:**

    * Relative error is ``O(1/n)`` in moderate-deviation regions and
      degrades in the far tails (large deviations) and for small n —
      expect 1-2 significant digits fewer than the exact FFT path.
    * The lattice correction assumes span 1; if ``gcd(w) = h > 1`` the
      achievable sums sit on ``h * Z`` and the approximation smooths
      over gaps (error up to the local mass of one lattice cell).
    * Tails outside the support are handled exactly (returns 0.0 / 1.0
      once deterministic shifts are accounted for).
    """
    p, w, _ = validate_inputs(p, w)
    p_s, w_s, shift = _strip_degenerate(p, w)

    if p_s.size == 0:
        return 1.0 if s >= shift else 0.0

    s_eff = float(s) - shift
    support = float(w_s.sum())
    if s_eff < 0.0:
        return 0.0
    if s_eff >= support:
        return 1.0

    s_c = s_eff + 0.5  # continuity correction for lattice data
    mean = float(_kgf(0.0, p_s, w_s)[1])

    if abs(s_c - mean) < 1e-10:
        return 0.5  # limiting value at the mean

    t_hat = _solve_saddlepoint(s_c, p_s, w_s)
    K, _, K2 = _kgf(t_hat, p_s, w_s)

    w_lr = math.copysign(math.sqrt(2.0 * (t_hat * s_c - K)), t_hat)
    # Lattice-corrected second factor: (1 - e^{-t}) instead of t
    u_lr = (1.0 - math.exp(-t_hat)) * math.sqrt(K2)

    upper = 1.0 - _norm_cdf(w_lr) + _norm_pdf(w_lr) * (1.0 / w_lr - 1.0 / u_lr)
    return float(min(1.0, max(0.0, 1.0 - upper)))


# ---------------------------------------------------------------------------
# Percentile of an observed outcome
# ---------------------------------------------------------------------------

def percentile_of_outcome(
    p: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    method: str = "fft",
) -> float:
    """Percentile of an observed outcome on the WPB CDF.

    Parameters
    ----------
    p : (n,) array_like of float
        Per-trial success probabilities.
    w : (n,) array_like of int
        Per-trial positive integer weights.
    x : (n,) array_like of bool
        Observed outcomes; the observed weighted sum is
        ``s_obs = sum(w_i * x_i)``.
    method : {"fft", "saddlepoint"}, default "fft"
        ``"fft"`` evaluates the exact CDF (recommended for moderate
        supports); ``"saddlepoint"`` uses the Lugannani-Rice
        approximation with ``O(n)`` memory — see
        :func:`saddlepoint_cdf` for accuracy trade-offs.

    Returns
    -------
    float
        ``P(S <= s_obs)`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        On invalid inputs or an unknown ``method``.
    """
    p, w, x = validate_inputs(p, w, x)
    s_obs = int(np.dot(w, x))  # zero-weight trials contribute 0 exactly

    if method == "fft":
        cdf = wpb_cdf(p, w)
        return float(cdf[min(s_obs, cdf.shape[0] - 1)])
    if method == "saddlepoint":
        return saddlepoint_cdf(p, w, float(s_obs))
    raise ValueError(f"unknown method {method!r}; expected 'fft' or 'saddlepoint'")


if __name__ == "__main__":
    p
    
    per = percentile_of_outcome()