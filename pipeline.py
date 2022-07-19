# sys.path.append("~/.local/lib/python3.6/site-packages")
import healpy as hp
import numpy as np
import pylab as pl

# Reproduce Planck colormap
from matplotlib.colors import ListedColormap
'''
colombi1_cmap = ListedColormap(
   np.loadtxt("https://raw.githubusercontent.com/zonca/paperplots/master/data/Planck_Parchment_RGB.txt") / 255.)
colombi1_cmap.set_bad("gray")  # color of missing pixels
colombi1_cmap.set_under("white")  # color of background, necessary if you want to use
cmap = colombi1_cmap

get_ipython().run_line_magic('matplotlib', 'inline')
'''
# In[265]:


t = 1.01
p0 = 0.22
p = 0.25
f = 0.75

# In[266]:


nside = 256

dust_map = hp.read_map('../data/COM_CompMap_ThermalDust-commander_2048_R2.00.fits', 0)

cst = 56.8
nu = 545.  # The reference frequency in GHz
CMBtoRJ = (nu / cst) ** 2 * np.exp(nu / cst) / (np.exp(nu / cst) - 1) ** 2

dust_map *= 1. / CMBtoRJ

dust_map = hp.ud_grade(dust_map, nside)


def AtoB(alm):
    blm = alm
    blm[0] = t * alm[0]
    blm[1] = alm[1] + p * alm[0]
    blm[2] = p0 * f * alm[2] / p
    return blm


# In[268]:


def do_projection(B, Bdims='sp'):
    B = B / np.sqrt(np.einsum('%s,%s->p' % (Bdims, Bdims), B, B))
    Bperp = B - r * np.einsum('%s,sp->p' % Bdims, B, r)
    phi = np.pi - np.sign(np.einsum('sp,sp->p', Bperp, e)) * np.arccos(
        np.einsum('sp,sp->p', Bperp, n) / np.sqrt(np.einsum('sp,sp->p', Bperp, Bperp)))
    cos2gamma = 1. - np.abs(np.einsum('%s,sp->p' % Bdims, B, r)) ** 2
    return phi, cos2gamma


# In[269]:


lmax = 3 * nside - 1
ell = np.arange(lmax + 1)
npix = hp.nside2npix(nside)
npix = hp.nside2npix(nside)
r = hp.pix2vec(nside, np.arange(npix))
# North vector n=(rxz)xz (z=[0,0,1])
n = r * r[2] * (-1.)
n[2] += 1.
n = n / np.sqrt(np.einsum('sp,sp->p', n, n))

# East vector e=-rxn
e = np.cross(r, n, axisa=0, axisb=0).T * (-1.)
e = e / np.sqrt(np.einsum('sp,sp->p', e, e))


def simulate_GMF(l0, b0, p0, alphaM, fM, N):
    N = int(N)

    B0 = np.array([np.cos(l0) * np.cos(b0), np.sin(l0) * np.cos(b0), np.sin(b0)])

    phi = np.zeros((N, npix))
    cos2gamma = np.zeros((N, npix))

    Cell = ell ** alphaM
    Cell[:2] = 0.

    for i in range(N):
        while True:
            Bt = np.array([hp.synfast(Cell, nside, verbose=False), hp.synfast(Cell, nside, verbose=False),
                           hp.synfast(Cell, nside, verbose=False)])
            Bt = Bt / np.sqrt(np.einsum('sp,sp->p', Bt, Bt))
            B = B0[:, np.newaxis] + fM * Bt
            phi[i], cos2gamma[i] = do_projection(B)
            if (np.sum(np.isnan(phi[i])) + np.sum(np.isnan(cos2gamma[i]))) == 0: break

    return cos2gamma, phi


# In[270]:


d2r = np.pi / 180.


def make_dustsim_new(dustMap, l0=70 * d2r, b0=24 * d2r, p0=0.25, alphaM=-2.5, fM=0.9, N=7):
    dust_sim = np.zeros((3, npix))

    cos2gamma, phi = simulate_GMF(l0, b0, p0, alphaM, fM, N)

    S = dustMap / np.sum(1. - (cos2gamma - 2. / 3.) * p0, axis=0)

    dust_sim[0] = dustMap
    dust_sim[1] = S * p0 * np.sum(np.cos(2. * phi) * cos2gamma, axis=0)
    dust_sim[2] = S * p0 * np.sum(np.sin(2. * phi) * cos2gamma, axis=0)

    return dust_sim


# In[271]:


dust_sim = make_dustsim_new(dust_map)

# In[272]:

import pysm3 as pysm
import pysm.units as u

def freq_scaling(inmap, infreq, outfreq):
    sky = pysm.Sky(nside=nside, preset_strings=["d0"])
    sky.components[0].I_ref = (inmap[0] * u.uK_CMB).to("uK_RJ", equivalencies=u.cmb_equivalencies(infreq * u.GHz))
    sky.components[0].Q_ref = (inmap[1] * u.uK_CMB).to("uK_RJ", equivalencies=u.cmb_equivalencies(infreq * u.GHz))
    sky.components[0].U_ref = (inmap[2] * u.uK_CMB).to("uK_RJ", equivalencies=u.cmb_equivalencies(infreq * u.GHz))
    sky.components[0].freq_ref_I = infreq * u.GHz
    sky.components[0].freq_ref_P = infreq * u.GHz

    outmap = sky.get_emission(outfreq * u.GHz)
    outmap = outmap.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(outfreq * u.GHz))
    outmap = outmap.value

    return outmap


# In[273]:


model_f353_map = freq_scaling(dust_sim, 545, 353)


# In[274]:


def CtoD(cl, lmax):
    l = np.arange(lmax + 1)
    l = np.arange(lmax + 1)
    return l * (l + 1) / 2 / np.pi * cl


# In[275]:


def averaging(start, end, dl):
    r = [0, 0, 0]
    r[0] = np.sum((dl[0][start:end + 1] / dl[1][start:end + 1])) / (end - start + 1)
    r[1] = np.sum((dl[3][start:end + 1] / dl[1][start:end + 1])) / (end - start + 1)
    r[2] = np.sum((dl[2][start:end + 1] / dl[1][start:end + 1])) / (end - start + 1)
    return r


# In[276]:


f100_map = hp.read_map('data/HFI_SkyMap_100_2048_R4.00_full.fits', [0, 1, 2])
# f100_map[0]=hp.remove_dipole(f100_map[0],gal_cut=30) # necessary for visualizing this specific map, but details are irrelevant now

f100_map = hp.ud_grade(f100_map, nside)
f100_map *= 1e6  # These maps are given in K_{CMB}d

f353_map = hp.read_map('data/HFI_SkyMap_353_2048_R4.00_full.fits', (0, 1, 2))
# f353_map[0]=hp.remove_dipole(f353_map[0],gal_cut=30) # necessary for visualizing this specific map, but details are irrelevant now

f353_map = hp.ud_grade(f353_map, nside)
f353_map *= 1e6  # These maps are given in K_{CMB}

rot = hp.Rotator(coord=['C', 'G'])

mask = hp.read_map('data/bk14_mask_cel_n0512.fits')
mask = np.nan_to_num(hp.ud_grade(mask, nside))
mask[mask < 0] = 0
pl.figure(figsize=(12, 4))

hp.mollview(mask, sub=(1, 2, 1), title='celestial coordinates', coord='C')
hp.graticule(coord='G', color='white')  # Shows coordinate grid in Galactic coordinates

mask = rot.rotate_map_pixel(mask)
hp.mollview(mask, sub=(1, 2, 2), title='Galactic coordinates', coord='G')
hp.graticule(coord='G', color='white')  # Shows coordinate grid in Galactic coordinates

f353_map_1st = hp.read_map('data/HFI_SkyMap_353_2048_R4.00_full-ringhalf-1.fits', [0, 1, 2])
f353_map_1st = hp.ud_grade(f353_map_1st, nside)
f353_map_1st *= 1e6  # These maps are given in K_{CMB}

f353_map_2nd = hp.read_map('data/HFI_SkyMap_353_2048_R4.00_full-ringhalf-2.fits', [0, 1, 2])
f353_map_2nd = hp.ud_grade(f353_map_2nd, nside)
f353_map_2nd *= 1e6  # These maps are given in K_{CMB}

# In[277]:


mask2 = hp.read_map('data/HFI_Mask_GalPlane-apo2_2048_R2.00.fits')
mask2 = np.nan_to_num(hp.ud_grade(mask2, nside))
mask2[mask2 < 0] = 0
pl.figure(figsize=(12, 4))

hp.mollview(mask2, sub=(1, 2, 1), title='celestial coordinates', coord='C')
hp.graticule(coord='G', color='white')  # Shows coordinate grid in Galactic coordinates

hp.mollview(mask2, sub=(1, 2, 2), title='Galactic coordinates', coord='G')
hp.graticule(coord='G', color='white')  # Shows coordinate grid in Galactic coordinates

# In[278]:


lmax = 500

fsky = (np.mean(mask ** 2))

cl_real = hp.anafast(mask * f353_map, lmax=lmax) / fsky
cl_sim = hp.anafast(mask2 * model_f353_map, lmax=lmax) / fsky

cl_1st = hp.anafast(mask * f353_map_1st, lmax=lmax) / fsky
cl_2nd = hp.anafast(mask * f353_map_2nd, lmax=lmax) / fsky
cl_cross = hp.anafast(mask * f353_map_1st, map2=mask * f353_map_2nd, lmax=lmax) / fsky
cl_new = hp.anafast(mask2 * f353_map, lmax=lmax) / fsky


# In[279]:


def CtoR(cl, start=60, end=100):
    return averaging(start, end, CtoD(cl, lmax))


# In[280]:


print(CtoR(cl_new, 60, 100))

# In[281]:


B
to
E
should
be
approximately
0.5


# In[282]:


def plotspectra(cl_new):
    l = np.arange(lmax + 1)
    pl.figure(figsize=(18, 4.5))

    ax = pl.subplot(1, 3, 1)
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_new[0])
    ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
    ax.set_ylabel('$D_{\ell}^{TT}$ $[\mu K^2]$', fontsize=14)

    ax = pl.subplot(1, 3, 2)
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_new[1])
    ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
    ax.set_ylabel('$D_{\ell}^{EE}$ $[\mu K^2]$', fontsize=14)

    ax = pl.subplot(1, 3, 3)
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_new[2])

    ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
    ax.set_ylabel('$D_{\ell}^{BB}$ $[\mu K^2]$', fontsize=14)


# In[283]:


alm = hp.map2alm(model_f353_map)

# In[284]:


blm = AtoB(alm)

# In[312]:


new_map = hp.alm2map(blm, nside)
pl.figure(figsize=(12, 8))
hp.mollview(new_map[0], title='model T', min=-500, max=1265, cmap=cmap, unit=r'$\mu K_{CMB}$', norm='hist',
            sub=(2, 2, 1))
# hp.mollview(new_map[1],title='model Q', min = cmap=cmap, unit=r'$\mu K_{CMB}$', norm='hist', sub=(4,2,1))
# hp.mollview(new_map[2],title='model U', cmap=cmap, unit=r'$\mu K_{CMB}$', norm='hist', sub=(4,2,2))

hp.mollview(mask2 * model_f353_map[0], title='model T', min=-500, max=1265, cmap=cmap, unit=r'$\mu K_{CMB}$',
            norm='hist', sub=(2, 2, 2))
# hp.mollview(mask2 * model_f353_map[1],title='model Q',  cmap=cmap, unit=r'$\mu K_{CMB}$', norm='hist', sub=(4,2,4))
# hp.mollview(mask2 * model_f353_map[2],title='model U', cmap=cmap, unit=r'$\mu K_{CMB}$', norm='hist', sub=(4,2,5))
hp.mollview(mask2 * f353_map[0], title='model T', min=-1265, max=-100, cmap=cmap, unit=r'$\mu K_{CMB}$', norm='hist',
            sub=(2, 2, 3))

cl_v = hp.anafast(mask2 * new_map, lmax=lmax) / fsky
# plotspectra(cl_v)
print(CtoR(cl_v, 60, 100))

new_cl_cross = hp.anafast(mask2 * f353_map, map2=mask2 * new_map, lmax=lmax) / fsky
l = np.arange(new_cl_cross.shape[-1])
pl.figure(figsize=(6, 4.5))
pl.plot(l, l * (l + 1) / 2 / np.pi * new_cl_cross[0], label="TT")
# pl.plot(l,l*(l+1)/2/np.pi*new_cl_cross[1],label = "EE")
# pl.plot(l,l*(l+1)/2/np.pi*new_cl_cross[2],label = "BB")
# pl.plot(l,l*(l+1)/2/np.pi*new_cl_cross[3], label = "TE")
pl.legend(fontsize=14)

# Ask Dominic what the polarization angle dispersion function is
#

# In[313]:


new_map = hp.alm2map(blm, nside)
cl_v_small = hp.anafast(mask * new_map, lmax=lmax) / fsky
plotspectra(cl_v)
print(CtoR(cl_v_small, 60, 100))

# In[314]:


l = np.arange(lmax + 1)
pl.figure(figsize=(24, 4.5))

ax = pl.subplot(1, 4, 1)
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v[0])
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v_small[0])

ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
ax.set_ylabel('$D_{\ell}^{TT}$ $[\mu K^2]$', fontsize=14)

ax = pl.subplot(1, 4, 2)
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v[1])
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v_small[2])

ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
ax.set_ylabel('$D_{\ell}^{EE}$ $[\mu K^2]$', fontsize=14)

ax = pl.subplot(1, 4, 3)
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v[2])
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v_small[1])

ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
ax.set_ylabel('$D_{\ell}^{BB}$ $[\mu K^2]$', fontsize=14)

ax = pl.subplot(1, 4, 4)
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v[3])
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_v_small[3])

ax.set_xlabel('Multipole moment $\ell$', fontsize=14)
ax.set_ylabel('$D_{\ell}^{TE}$ $[\mu K^2]$', fontsize=14)


# In[326]:


def singleplot(cl):
    l = np.arange(cl.shape[-1])
    pl.figure(figsize=(6, 4.5))
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl[0], label="TT")
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl[1], label="EE")
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl[2], label="BB")
    pl.loglog(l, l * (l + 1) / 2 / np.pi * cl[3], label="TE")
    pl.legend(fontsize=14)
    # ax.set_xlabel('Multipole moment $\ell$',fontsize=14)


# In[327]:


singleplot(cl_v)

# In[328]:


singleplot(cl_sim)

# In[329]:


singleplot(cl_v_small)

# In[330]:


singleplot(cl_real)

# In[331]:


singleplot(cl_new)

# In[338]:


bins = np.arange(30, 200, 10)
b = .5 * (bins[1:] + bins[:-1])


def bin_spectra(cl):
    cb = np.zeros((*np.shape(cl)[:-1], len(bins) - 1))
    for i in range(len(bins) - 1):
        cb[..., i] = np.mean(cl[..., bins[i]:bins[i + 1]], axis=-1)
    return cb


bincl_v = bin_spectra(l * (l + 1) / 2 / np.pi * cl_v)
pl.figure(figsize=(6, 4.5))
pl.loglog(b, bincl_v[0], label="TT")
pl.loglog(b, bincl_v[1], label="EE")
pl.loglog(b, bincl_v[2], label="BB")
pl.loglog(b, bincl_v[3], label="TE")
pl.legend(fontsize=14)
# ax.set_xlabel('Multipole moment $\ell$',fontsize=14)


# In[339]:


print(bin_spectra(cl_v).shape)
singleplot(bin_spectra(cl_v))

# In[340]:

bincl_cross = bin_spectra(l * (l + 1) / 2 / np.pi * cl_cross)
pl.figure(figsize=(24, 4.5))
ax = pl.subplot(1, 4, 1)
pl.loglog(b, bincl_cross[0], label="TT_real")
pl.loglog(b, bincl_v[0], label="TT")
pl.legend(fontsize=14)
ax = pl.subplot(1, 4, 2)
pl.loglog(b, bincl_cross[1], label="EE_real")
pl.loglog(b, bincl_v[1], label="EE")
pl.legend(fontsize=14)
ax = pl.subplot(1, 4, 3)
pl.loglog(b, bincl_cross[2], label="BB_real")
pl.loglog(b, bincl_v[2], label="BB")
pl.legend(fontsize=14)
ax = pl.subplot(1, 4, 4)
pl.loglog(b, bincl_cross[3], label="TE_real")
pl.loglog(b, bincl_v[3], label="TE")
pl.legend(fontsize=14)
# ax.set_xlabel('Multipole moment $\ell$',fontsize=14)e

f545_map = freq_scaling(f353_map, 353, 545)
cl_545 = hp.anafast(f545_map, lmax=lmax)
cl_dust = hp.anafast(dust_map, lmax=lmax)
pl.figure(figsize=(6, 4.5))
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_dust, label="TT")
pl.loglog(l, l * (l + 1) / 2 / np.pi * cl_545[0], label="TT")
'''