"""|F[y'']|^2, its peak and the left-sided FWHM, against analytic spectra.

For y = e^{-g t} cos(w0 t), y'' is a damped cosine with the same g and w0, whose
power spectrum near w0 is a Lorentzian ~ 1 / ((w - w0)^2 + g^2): peak at w0,
FWHM = 2 g. A pure decay y = e^{-g t} peaks at w = 0 with the same FWHM, read
on the mirrored side.

Run directly (``python tests/test_y2_spectrum.py``) or under pytest.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "src", "intracellular_transport")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from computational_tools import y2_spectrum as y2s   # noqa: E402

H, T = 1e-3, 20.0
t = np.arange(int(T / H) + 1) * H


def test_damped_cosine_peak_and_fwhm():
    g, w0 = 2.0, 60.0
    spec = y2s.spectrum(np.exp(-g * t) * np.cos(w0 * t), H)
    assert abs(spec["omega_peak"] - w0) < 0.05, spec["omega_peak"]
    assert abs(spec["fwhm"] - 2 * g) < 0.02 * 2 * g, spec["fwhm"]
    assert spec["fwhm_side"] == "left"
    assert spec["omega_left"] < spec["omega_peak"]


def test_peak_at_zero_reads_the_mirrored_side():
    g = 3.0
    spec = y2s.spectrum(np.exp(-g * t), H)
    assert spec["omega_peak"] == 0.0
    assert spec["fwhm_side"] == "mirrored"
    assert abs(spec["fwhm"] - 2 * g) < 0.02 * 2 * g, spec["fwhm"]


def test_right_flank_leakage_does_not_widen_the_fwhm():
    """A weaker, broad neighbour on the right lifts that flank; the left half-width is untouched."""
    g, w0 = 2.0, 60.0
    alone = y2s.spectrum(np.exp(-g * t) * np.cos(w0 * t), H)
    mixed = y2s.spectrum(np.exp(-g * t) * np.cos(w0 * t)
                         + 0.02 * np.exp(-8.0 * t) * np.cos(68.0 * t), H)
    assert abs(mixed["fwhm"] - alone["fwhm"]) < 0.02 * alone["fwhm"]


def test_trim_keeps_the_peak():
    spec = y2s.spectrum(np.exp(-2.0 * t) * np.cos(60.0 * t), H)
    small = y2s.trim(spec)
    assert small["omega"].size < spec["omega"].size
    assert small["omega"][0] <= spec["omega_peak"] <= small["omega"][-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
