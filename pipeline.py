# sys.path.append("~/.local/lib/python3.6/site-packages")
import healpy as hp
import numpy as np
import pylab as pl

# Reproduce Planck colormap
from matplotlib.colors import ListedColormap

colombi1_cmap = ListedColormap(
   np.loadtxt("https://raw.githubusercontent.com/zonca/paperplots/master/data/Planck_Parchment_RGB.txt") / 255.)
colombi1_cmap.set_bad("gray")  # color of missing pixels
colombi1_cmap.set_under("white")  # color of background, necessary if you want to use
cmap = colombi1_cmap

get_ipython().run_line_magic('matplotlib', 'inline')

# In[265]:


t = 1.01
p0 = 0.22
p = 0.25
f = 0.75

def AtoB(alm):
    blm = np.zeros(alm.shape, dtype = complex)
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

nside = 256
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





d2r = np.pi / 180.


def make_dustsim_new(dustMap, l0=70 * d2r, b0=24 * d2r, p0=0.25, alphaM=-2.5, fM=0.9, N=7):
    dust_sim = np.zeros((3, npix))

    cos2gamma, phi = simulate_GMF(l0, b0, p0, alphaM, fM, N)

    S = dustMap / np.sum(1. - (cos2gamma - 2. / 3.) * p0, axis=0)

    dust_sim[0] = dustMap
    dust_sim[1] = S * p0 * np.sum(np.cos(2. * phi) * cos2gamma, axis=0)
    dust_sim[2] = S * p0 * np.sum(np.sin(2. * phi) * cos2gamma, axis=0)

    return dust_sim




import pysm3 as pysm
import pysm3.units as u

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


def CtoD(cl, lmax):
    l = np.arange(lmax + 1)
    l = np.arange(lmax + 1)
    return l * (l + 1) / 2 / np.pi * cl


def averaging(start, end, dl):
    r = [0, 0, 0]
    r[0] = np.sum((dl[0][start:end + 1] / dl[1][start:end + 1])) / (end - start + 1)
    r[1] = np.sum((dl[3][start:end + 1] / dl[1][start:end + 1])) / (end - start + 1)
    r[2] = np.sum((dl[2][start:end + 1] / dl[1][start:end + 1])) / (end - start + 1)
    return r
   
   
   
def Vasyngel_update(m, nside):
    alm = hp.map2alm(m)
    blm = AtoB(alm)
    new_map = hp.alm2map(blm, nside)
    return new_map
   
def singleplot(cl):
    l = np.arange(cl.shape[-1])
    pl.figure(figsize=(6,4.5))
    pl.loglog(l,l*(l+1)/2/np.pi*cl[0],label = "TT")
    pl.loglog(l,l*(l+1)/2/np.pi*cl[1],label = "EE")
    pl.loglog(l,l*(l+1)/2/np.pi*cl[2],label = "BB")
    #pl.loglog(l,l*(l+1)/2/np.pi*cl[3], label = "TE")
    pl.legend(fontsize=14)
    #ax.set_xlabel('Multipole moment $\ell$',fontsize=14)
    
def doubleplot(bincl_cross, bincl_v):
    b = np.arange(bincl_v.shape[-1])
    pl.figure(figsize=(24,4.5))
    ax=pl.subplot(1,4,1)
    pl.loglog(b,bincl_cross[0],label = "TT_real")
    pl.loglog(b,bincl_v[0],label = "TT")
    pl.legend(fontsize=14)
    ax=pl.subplot(1,4,2)
    pl.loglog(b,bincl_cross[1],label = "EE_real")
    pl.loglog(b,bincl_v[1],label = "EE")
    pl.legend(fontsize=14)
    ax=pl.subplot(1,4,3)
    pl.loglog(b,bincl_cross[2],label = "BB_real")
    pl.loglog(b,bincl_v[2],label = "BB")
    pl.legend(fontsize=14)
    ax=pl.subplot(1,4,4)
    pl.loglog(b,bincl_cross[3],label = "TE_real")
    pl.loglog(b,bincl_v[3],label = "TE")
    pl.legend(fontsize=14)
