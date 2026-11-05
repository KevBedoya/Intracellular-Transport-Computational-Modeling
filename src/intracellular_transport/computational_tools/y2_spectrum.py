"""Power spectrum of y'' = d^2 ln M / dt^2, its peak, and a left-sided FWHM.

    P(w) = | H sum_n y''(t_n) exp(-i w t_n) |^2,     t_n = n H,

the squared magnitude of the Riemann-sum Fourier transform of y'' over the
record, with no window and no detrending -- the convention of
results/2026-10-07_ab-grid-96x96_tau0.01/scripts/psd_96.py. w is angular
frequency (rad per unit t).

y'' is the central difference of y = ln M at spacing H (~1e-4, the t*
criterion's CHAR_TIME_DIFF_SPACING), whichever engine produced M. The modal
engine could give y'' exactly at each t_n, but the early pulse of y'' is far
narrower than H: point samples of it would put the wrong weight into the
low-frequency bins, where the peak is, while the central difference carries
its integral. So both engines are transformed the same way.

The record is zero-padded to PAD_FACTOR times its length, which samples the
same transform PAD_FACTOR times more finely than the native bin spacing 2 pi / T
(it adds no resolution, but the peak and the half-maximum crossing are no
longer snapped to those bins).

FWHM. The right flank of the peak can carry leakage from other modes, so the
half-width is read on the left only and doubled:

    FWHM = 2 (w_peak - w_left),   P(w_left) = P(w_peak) / 2,  w_left < w_peak.

P is even in w (y'' is real), so when the curve stays above half-maximum all
the way down to w = 0, the left crossing lies on the mirrored side, at
w_left = -w_c with w_c the first half-maximum crossing right of the peak; a
peak at w = 0 is the case w_peak = 0, FWHM = 2 w_c. ``fwhm_side`` records which
("left" or "mirrored").
"""
import numpy as np

PAD_FACTOR = 8
ZOOM_HALF_WIDTHS = 4          # zoomed plots span the peak +- this many half-widths
PSD_COLUMNS = ("psd_omega_peak", "psd_peak", "psd_fwhm", "psd_fwhm_side")


def spacing(sample_dt, target):
    """(stride s, H = s * sample_dt): the multiple of the sample spacing nearest ``target``."""
    s = max(1, int(round(target / sample_dt)))
    return s, s * sample_dt


def _crossing(w0, p0, w1, p1, level):
    return w0 + (level - p0) * (w1 - w0) / (p1 - p0)


def spectrum(y, H, pad=PAD_FACTOR):
    """P(w) of the central-difference y'' of ``y`` (ln M at t_n = n H), its peak and FWHM.

    Only the leading run of finite ``y`` is used. Returns a dict: ``omega``,
    ``psd``, ``H``, ``d_omega`` (native bin spacing 2 pi / T), ``omega_peak``,
    ``psd_peak``, ``omega_left``, ``fwhm`` and ``fwhm_side``; the last four are
    NaN / "none" when the record is too short or no half-maximum crossing exists.
    """
    y = np.asarray(y, dtype=np.float64)
    bad = np.nonzero(~np.isfinite(y))[0]
    if bad.size:
        y = y[:bad[0]]
    nan = float("nan")
    out = dict(omega=np.zeros(0), psd=np.zeros(0), H=float(H), d_omega=nan,
               omega_peak=nan, psd_peak=nan, omega_left=nan, fwhm=nan, fwhm_side="none")
    if y.size < 5:
        return out

    d2 = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / H ** 2
    n = pad * d2.size
    w = 2.0 * np.pi * np.fft.rfftfreq(n, H)
    P = np.abs(H * np.fft.rfft(d2, n)) ** 2
    k = int(np.argmax(P))
    half = 0.5 * P[k]

    below = np.nonzero(P[:k + 1] < half)[0]
    if below.size:
        j = int(below[-1])
        w_left, side = _crossing(w[j], P[j], w[j + 1], P[j + 1], half), "left"
    else:
        right = np.nonzero(P[k:] < half)[0]
        if right.size:
            j = k + int(right[0])
            w_left, side = -_crossing(w[j - 1], P[j - 1], w[j], P[j], half), "mirrored"
        else:
            w_left, side = nan, "none"

    out.update(omega=w, psd=P, d_omega=2.0 * np.pi / (d2.size * H),
               omega_peak=float(w[k]), psd_peak=float(P[k]), omega_left=float(w_left),
               fwhm=float(2.0 * (w[k] - w_left)), fwhm_side=side)
    return out


def summary(spec):
    """The scalar columns of ``spec``, keyed by PSD_COLUMNS."""
    return dict(psd_omega_peak=spec["omega_peak"], psd_peak=spec["psd_peak"],
                psd_fwhm=spec["fwhm"], psd_fwhm_side=spec["fwhm_side"])


def _window(spec):
    w, wp, fwhm = spec["omega"], spec["omega_peak"], spec["fwhm"]
    half_width = 0.5 * fwhm if np.isfinite(fwhm) and fwhm > 0 else 10 * spec["d_omega"]
    lo = max(0.0, wp - ZOOM_HALF_WIDTHS * half_width)
    hi = wp + ZOOM_HALF_WIDTHS * half_width
    return (w >= lo) & (w <= hi), lo, hi


def trim(spec):
    """``spec`` with omega and psd cut to the zoom window: what a plot needs, small to pickle."""
    if not spec["omega"].size:
        return spec
    keep, _, _ = _window(spec)
    return dict(spec, omega=spec["omega"][keep], psd=spec["psd"][keep])


def plot_peak(ax, spec, title):
    """P(w) about its peak, linear axes: the peak, the half maximum and the FWHM."""
    if not spec["omega"].size or not np.isfinite(spec["psd_peak"]):
        ax.text(0.5, 0.5, "no spectrum", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return
    keep, lo, hi = _window(spec)
    wp, pp, fwhm = spec["omega_peak"], spec["psd_peak"], spec["fwhm"]
    ax.plot(spec["omega"][keep], spec["psd"][keep], lw=1.0, color="C0")
    ax.axvline(wp, color="red", lw=1.0)
    ax.axhline(0.5 * pp, color="0.4", lw=0.8, ls=":")
    if np.isfinite(fwhm):
        # the measured left half-width, and its mirror that completes the FWHM
        ax.axvspan(wp - 0.5 * fwhm, wp + 0.5 * fwhm, color="C2", alpha=0.12, lw=0)
        if spec["omega_left"] >= lo:
            ax.axvline(spec["omega_left"], color="C2", lw=1.0, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.text(0.98, 0.96,
            rf"$\omega_{{peak}}$ = {wp:.4g}" + "\n" + f"peak = {pp:.4g}" + "\n"
            + f"FWHM = {fwhm:.4g} ({spec['fwhm_side']})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))


def save_figure(path, specs, titles, ncols, suptitle):
    """One zoomed panel per spectrum, ``ncols`` to a row, saved to ``path``."""
    import matplotlib.pyplot as plt

    nrows = max(1, -(-len(specs) // ncols))
    # at least 7 in wide, so the two-line title fits over a single panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(4.0 * ncols, 7.0), 3.0 * nrows + 0.6),
                             squeeze=False)
    for ax in axes.flat[len(specs):]:
        ax.set_visible(False)
    for ax, spec, title in zip(axes.flat, specs, titles):
        plot_peak(ax, spec, title)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\omega$ (rad / unit t)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|\mathcal{F}[y'']|^2$")
    fig.suptitle(suptitle + "\n" + r"$y = \ln M$; red = peak, shaded = FWHM "
                 r"= 2 $\times$ left half-width (dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
