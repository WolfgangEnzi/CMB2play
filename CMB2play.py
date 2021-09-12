"""
================================================================================

CMB2play v1.0 - Wolfgang Enzi 2021

A small python script to illustrate the how cosmological parameters affect the
CMB. The user can modify certain cosmological parameters and see their effect
on a random realization of the CMB. The idea is based on the
"Build a Universe!" application,
https://wmap.gsfc.nasa.gov/resources/camb_tool/index.html,
which can no longer be used after flash player has expired.

Requirements:

- the python package of CAMB, which can be found at
https://camb.readthedocs.io/en/latest/ and http://camb.info.

- the python package of Healpix, which can be found at
https://healpy.readthedocs.io/en/latest/.

I recommend to create a new python environment before installing camb and
healpy with "pip install camb" and "pip install healpy", it may request
specific versions of standard packages.

================================================================================
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import healpy as hp
import camb


Nll = 1500
Seed = 532349862
np.random.seed(Seed)


def get_spec(OmC, OmB, OmL, h, zrei, ns):
    pars = camb.CAMBparams()
    OmK = 1 - OmB - OmC - OmL
    OmBh2 = OmB * h * h
    pars.set_cosmology(H0=h * 100, ombh2=OmBh2,
                       omch2=OmC * h * h, omk=OmK, zrei=zrei)
    pars.InitPower.set_params(As=2e-9, ns=ns, r=0)
    pars.set_for_lmax(Nll, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total'][:, 0]
    return np.arange(len(totCL)), totCL


fig = plt.figure(figsize=(9, 5))
plt.subplots_adjust(right=0.95, left=0.11, wspace=0.2,
                    bottom=0.1, hspace=0.25, top=0.95)
ax = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 3), (2, 0))
ax3 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=2)
cosmo_default = [0.25, 0.05, 0.7, 0.677, 7.82, 0.9665]
ll0, spec0 = get_spec(*cosmo_default)

maps = hp.sphtfunc.synfast(spec0 / (ll0 * (ll0 + 1) + 1e-30), 256, pol=False)
plt.axes(ax3)
axmol = hp.mollview(maps, title="", cbar=False, hold=True, cmap="jet")

l0, = ax.plot(ll0, spec0, lw=1, ls=":", color="r")
l, = ax.plot(ll0, spec0, lw=1, color="royalblue")
l02, = ax2.plot(ll0, np.ones(spec0.shape) - 1, lw=1, ls=":", color="r")
l2, = ax2.plot(ll0, np.ones(spec0.shape) - 1, lw=1, color="royalblue")
ax.set_ylim(0, 1.1 * np.max(spec0))
ax.set_xlim(10, np.max(ll0))
ax.set_xscale("log")
ax2.set_xscale("log")
ax2.set_xlim(10, np.max(ll0))
ax2.set_ylim(-1.3, 1.3)
ax2.set_xlabel("l")
ax2.set_ylabel(r"$\Delta C_l / C_l$")
ax.set_ylabel(r"$l(l+1)C_l$")

h_slider_ax = plt.axes([0.45, 0.05, 0.45, 0.02])
OmM_slider_ax = plt.axes([0.45, 0.1, 0.45, 0.02])
OmB_slider_ax = plt.axes([0.45, 0.15, 0.45, 0.02])
OmL_slider_ax = plt.axes([0.45, 0.2, 0.45, 0.02])
zrei_slider_ax = plt.axes([0.45, 0.25, 0.45, 0.02])
ns_slider_ax = plt.axes([0.45, 0.30, 0.45, 0.02])

OmC_slider = Slider(OmM_slider_ax,  r'$\Omega_{\rm cdm}$', 0.0, 1.0,
                    valinit=0.25, valstep=0.01, color="royalblue")
OmB_slider = Slider(OmB_slider_ax,  r'$\Omega_{\rm b}$', 0.01, 1.0,
                    valinit=0.05, valstep=0.01, color="royalblue")
h_slider = Slider(h_slider_ax, 'h', 0.0, 2.0,
                  valinit=0.677, valstep=0.01, color="royalblue")
OmL_slider = Slider(OmL_slider_ax, r'$\Omega_{\Lambda}$', 0.0, 1.0,
                    valinit=0.7, valstep=0.01, color="royalblue")
zrei_slider = Slider(zrei_slider_ax, r'$z_{\rm reion}$', 1, 50,
                     valinit=7.82, valstep=0.01, color="royalblue")
ns_slider = Slider(ns_slider_ax, r'$n_{\rm s}$', 0, 2,
                   valinit=0.9665, valstep=0.0001, color="royalblue")


def update(val):
    OmC = OmC_slider.val
    OmB = OmB_slider.val
    OmL = OmL_slider.val
    zrei = zrei_slider.val
    h = h_slider.val
    ns = ns_slider.val
    ll, spectrum = get_spec(OmC, OmB, OmL, h, zrei, ns)
    np.random.seed(Seed)
    maps = hp.sphtfunc.synfast(spectrum / (ll * (ll+1) + 1e-30),
                               256, pol=False)
    ax3 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=2)
    plt.axes(ax3)
    hp.mollview(maps, title="", cbar=False, hold=True, cmap="jet")
    u = min(len(ll), len(ll0))
    l.set_ydata(spectrum)
    l.set_xdata(ll)
    l2.set_ydata(spectrum[10:u] / spec0[10:u] - 1)
    l2.set_xdata(ll0[10:u])
    ax.set_ylim(0, 1.1 * np.max(np.append(spectrum[10:], spec0[10:]) - 1))
    v = max(1, np.max(np.fabs(spectrum[10:u] / spec0[10:u] - 1)))
    ax2.set_ylim(-1.3 * v, 1.3 * v)
    fig.canvas.draw_idle()


resetax = plt.axes([0.58, 0.35, 0.2, 0.04])
button = Button(resetax, 'Evaluate')
button.on_clicked(update)
plt.show()
