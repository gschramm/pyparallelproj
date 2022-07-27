# script to investigate the convergence of SPDHG for 2D TOF PET

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.algorithms import osem, spdhg
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom

import numpy as np
from scipy.ndimage import gaussian_filter

from pymirc.image_operations import grad, div

import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',
                    help='counts to simulate',
                    default=1e6,
                    type=float)
parser.add_argument('--niter',
                    help='number of iterations',
                    default=100,
                    type=int)
parser.add_argument('--niter_ref',
                    help='number of ref iterations',
                    default=5000,
                    type=int)
parser.add_argument('--nsubsets',
                    help='number of subsets',
                    default=28,
                    type=int)
parser.add_argument('--fwhm_mm',
                    help='psf modeling FWHM mm',
                    default=4.5,
                    type=float)
parser.add_argument('--fwhm_data_mm',
                    help='psf for data FWHM mm',
                    default=4.5,
                    type=float)
parser.add_argument('--ps_fwhm_mm',
                    help='FWHM mm of Gaussian for post-smoothing',
                    default=8.,
                    type=float)
parser.add_argument('--phantom', help='phantom to use', default='brain2d')
parser.add_argument('--seed',
                    help='seed for random generator',
                    default=1,
                    type=int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts = args.counts
niter = args.niter
niter_ref = args.niter_ref
nsubsets = args.nsubsets
fwhm_mm = args.fwhm_mm
fwhm_data_mm = args.fwhm_data_mm
phantom = args.phantom
seed = args.seed
ps_fwhm_mm = args.ps_fwhm_mm
beta = 0

gammas = np.array([0.3, 1., 3., 10., 30., 100., 300.])

it_early = 10
#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16, 1]),
                                       nmodules=np.array([28, 1]))

# setup a test image
voxsize = np.array([2., 2., 2.])
n2 = max(1, int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# sigma for post_smoothing
sig = ps_fwhm_mm / (voxsize * 2.35)

# convert fwhm from mm to pixels
fwhm = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
    n = 200
    img = np.zeros((n, n, n2), dtype=np.float32)
    tmp = ellipse2d_phantom(n=n, c=3)
    for i2 in range(n2):
        img[:, :, i2] = tmp
elif phantom == 'brain2d':
    n = 128
    img = np.zeros((n, n, n2), dtype=np.float32)
    tmp = brain2d_phantom(n=n)
    for i2 in range(n2):
        img[:, :, i2] = tmp

img_origin = (-(np.array(img.shape) / 2) + 0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins=17, tofbin_width=15.)
proj = ppp.SinogramProjector(scanner,
                             sino_params,
                             img.shape,
                             nsubsets=nsubsets,
                             voxsize=voxsize,
                             img_origin=img_origin,
                             tof=True,
                             sigma_tof=60. / 2.35,
                             n_sigmas=3.)

# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype=np.float32)

# estimate the norm of the operator
test_img = np.random.rand(*img.shape)
for i in range(5):
    fwd = ppp.pet_fwd_model(test_img, proj, attn_sino, sens_sino, fwhm=fwhm)
    back = ppp.pet_back_model(fwd, proj, attn_sino, sens_sino, fwhm=fwhm)

    pnsq = np.linalg.norm(back)
    print(i, np.sqrt(pnsq))

    test_img = back / pnsq

# normalize sensitivity sinogram to get PET forward model for 1 view with norm 1
# this is important otherwise the step size T in SPDHG get dominated by the gradient
sens_sino /= (np.sqrt(pnsq) / np.sqrt(8))

# forward project the image
img_fwd = ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, fwhm=fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
    scale_fac = (counts / img_fwd.sum())
    img_fwd *= scale_fac
    img *= scale_fac

    # contamination sinogram with scatter and randoms
    # useful to avoid division by 0 in the ratio of data and exprected data
    contam_sino = np.full(img_fwd.shape,
                          0.75 * img_fwd.mean(),
                          dtype=np.float32)

    em_sino = np.random.poisson(img_fwd + contam_sino)
else:
    scale_fac = 1.

    # contamination sinogram with sctter and randoms
    # useful to avoid division by 0 in the ratio of data and exprected data
    contam_sino = np.full(img_fwd.shape,
                          0.75 * img_fwd.mean(),
                          dtype=np.float32)

    em_sino = img_fwd + contam_sino

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# callback functions to calculate likelihood and show recon updates


def calc_cost(x):
    exp = ppp.pet_fwd_model(x, proj, attn_sino, sens_sino,
                            fwhm=fwhm) + contam_sino
    cost = (exp - em_sino * np.log(exp)).sum()

    return cost


def _cb(x, **kwargs):
    it = kwargs.get('iteration', 0)
    it_early = kwargs.get('it_early', -1)

    if it_early == it:
        if 'x_early' in kwargs:
            kwargs['x_early'][:] = x

    if 'cost' in kwargs:
        kwargs['cost'][it - 1] = calc_cost(x)
    if 'psnr' in kwargs:
        MSE = ((x - kwargs['xref'])**2).mean()
        kwargs['psnr'][it -
                       1] = 20 * np.log10(kwargs['xref'].max() / np.sqrt(MSE))
    if 'psnr_ps' in kwargs:
        x_ps = gaussian_filter(x, kwargs['sig'])
        MSE = ((x_ps - kwargs['xref_ps'])**2).mean()
        kwargs['psnr_ps'][it - 1] = 20 * np.log10(
            kwargs['xref_ps'].max() / np.sqrt(MSE))


#-----------------------------------------------------------------------------------------------
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('figs'):
    os.mkdir('figs')

# do long MLEM as reference
ref_fname = os.path.join(
    'data',
    f'mlem_{phantom}_niter_{niter_ref}_counts_{counts:.1E}_seed_{seed}.npz')

if os.path.exists(ref_fname):
    tmp = np.load(ref_fname)
    ref_recon = tmp['ref_recon']
    ref_cost = tmp['ref_cost']
else:
    ref_cost = np.zeros(niter_ref)

    proj.init_subsets(1)
    ref_recon = osem(em_sino,
                     attn_sino,
                     sens_sino,
                     contam_sino,
                     proj,
                     niter_ref,
                     fwhm=fwhm,
                     verbose=True,
                     callback=_cb,
                     callback_kwargs={'cost': ref_cost})
    proj.init_subsets(nsubsets)

    np.savez(ref_fname, ref_recon=ref_recon, ref_cost=ref_cost)

#----------

ref_recon_ps = gaussian_filter(ref_recon, sig)

proj.init_subsets(28)
init_recon = osem(em_sino,
                  attn_sino,
                  sens_sino,
                  contam_sino,
                  proj,
                  1,
                  fwhm=fwhm,
                  verbose=True)
proj.init_subsets(nsubsets)

cost_osem = np.zeros(niter)
psnr_osem = np.zeros(niter)
psnr_ps_osem = np.zeros(niter)

recon_osem_early = np.zeros(tuple(proj.img_dim), dtype=np.float32)

# do OSEM recon

cbk = {
    'cost': cost_osem,
    'xref': ref_recon,
    'psnr': psnr_osem,
    'xref_ps': ref_recon_ps,
    'psnr_ps': psnr_ps_osem,
    'sig': sig,
    'it_early': it_early,
    'x_early': recon_osem_early
}
recon_osem = osem(em_sino,
                  attn_sino,
                  sens_sino,
                  contam_sino,
                  proj,
                  niter,
                  fwhm=fwhm,
                  verbose=True,
                  xstart=init_recon,
                  callback=_cb,
                  callback_kwargs=cbk)

# SPHDG recon

ystart = 1 - (em_sino / (ppp.pet_fwd_model(
    init_recon, proj, attn_sino, sens_sino, fwhm=fwhm) + contam_sino))

costs_spdhg = np.zeros((len(gammas), niter))
psnr_spdhg = np.zeros((len(gammas), niter))
psnr_ps_spdhg = np.zeros((len(gammas), niter))

recons_spdhg = np.zeros((len(gammas), ) + img.shape, dtype=np.float32)
recons_spdhg_early = np.zeros((len(gammas), ) + img.shape, dtype=np.float32)

costs_spdhg2 = np.zeros((len(gammas), niter))
psnr_spdhg2 = np.zeros((len(gammas), niter))
psnr_ps_spdhg2 = np.zeros((len(gammas), niter))

recons_spdhg2 = np.zeros((len(gammas), ) + img.shape, dtype=np.float32)
recons_spdhg_early2 = np.zeros((len(gammas), ) + img.shape, dtype=np.float32)

for ig, gamma in enumerate(gammas):
    # SPDHG with warm start
    cbs = {
        'cost': costs_spdhg[ig, :],
        'xref': ref_recon,
        'psnr': psnr_spdhg[ig, :],
        'xref_ps': ref_recon_ps,
        'psnr_ps': psnr_ps_spdhg[ig, :],
        'sig': sig,
        'it_early': it_early,
        'x_early': recons_spdhg_early[ig, ...]
    }

    recons_spdhg[ig, ...] = spdhg(em_sino,
                                  attn_sino,
                                  sens_sino,
                                  contam_sino,
                                  proj,
                                  niter,
                                  gamma=gamma / img.max(),
                                  fwhm=fwhm,
                                  verbose=True,
                                  xstart=init_recon,
                                  ystart=ystart,
                                  callback=_cb,
                                  callback_kwargs=cbs)

    # SPDHG with cold start
    cbs2 = {
        'cost': costs_spdhg2[ig, :],
        'xref': ref_recon,
        'psnr': psnr_spdhg2[ig, :],
        'xref_ps': ref_recon_ps,
        'psnr_ps': psnr_ps_spdhg2[ig, :],
        'sig': sig,
        'it_early': it_early,
        'x_early': recons_spdhg_early2[ig, ...]
    }

    recons_spdhg2[ig, ...] = spdhg(em_sino,
                                   attn_sino,
                                   sens_sino,
                                   contam_sino,
                                   proj,
                                   niter,
                                   gamma=gamma / img.max(),
                                   fwhm=fwhm,
                                   verbose=True,
                                   callback=_cb,
                                   callback_kwargs=cbs2)

#-----------------------------------------------------------------------------------------------------

base_str = f'{phantom}_counts_{counts:.1E}_beta_{beta:.1E}_niter_{niter_ref}_{niter}_nsub_{nsubsets}'

ofile = os.path.join('figs', f'{base_str}.npz')

c0 = calc_cost(init_recon)
cref = ref_cost[-1]

# save the results
np.savez(ofile,
         ref_recon=ref_recon,
         ref_cost=ref_cost,
         recon_osem=recon_osem,
         cost_osem=cost_osem,
         recons_spdhg=recons_spdhg,
         costs_spdhg=costs_spdhg,
         c0=c0,
         cref=cref,
         gammas=gammas)

# show the relative cost and PSNR
it = np.arange(niter) + 1

fig2, ax2 = plt.subplots(3,
                         len(gammas),
                         figsize=(len(gammas) * 2, 6),
                         sharex=True,
                         sharey='row',
                         squeeze=False)

for ig, gamma in enumerate(gammas):
    ax2[0, ig].semilogy(it, (cost_osem - cref) / (c0 - cref),
                        label='OSEM',
                        color='tab:green')
    ax2[0, ig].semilogy(it, (costs_spdhg[ig, :] - cref) / (c0 - cref),
                        label=f'SPDHG warm',
                        color='tab:orange')
    ax2[0, ig].semilogy(it, (costs_spdhg2[ig, :] - cref) / (c0 - cref),
                        label=f'SPDHG cold',
                        color='tab:blue')
    ax2[0, ig].set_title(f'Gamma {gamma:.1E}', fontsize='medium')

    ax2[1, ig].plot(it, psnr_osem, label='OSEM', color='tab:green')
    ax2[1, ig].plot(it,
                    psnr_spdhg[ig, :],
                    label=f'SPDHG warm',
                    color='tab:orange')
    ax2[1, ig].plot(it,
                    psnr_spdhg2[ig, :],
                    label=f'SPDHG cold',
                    color='tab:blue')

    ax2[2, ig].plot(it, psnr_ps_osem, label='OSEM', color='tab:green')
    ax2[2, ig].plot(it,
                    psnr_ps_spdhg[ig, :],
                    label=f'SPDHG warm',
                    color='tab:orange')
    ax2[2, ig].plot(it,
                    psnr_ps_spdhg2[ig, :],
                    label=f'SPDHG cold',
                    color='tab:blue')
    ax2[2, 0].set_ylabel('PSNR ps')

ax2[0, 0].set_ylabel('relative cost')
ax2[1, 0].set_ylabel('PSNR')
ax2[0, 0].legend()

for axx in ax2[-1, :].flatten():
    axx.set_xlabel('iteration')

for axx in ax2.flatten():
    axx.grid(ls=':')

fig2.tight_layout()
fig2.savefig(os.path.join('figs', f'{base_str}_metrics.pdf'))
fig2.savefig(os.path.join('figs', f'{base_str}_metrics.png'))
fig2.show()

# show the reconstructions
fig3, ax3 = plt.subplots(4,
                         len(gammas) + 2,
                         figsize=((len(gammas) + 2) * 2, 8),
                         squeeze=False)

vmax = 1.5 * img.max()

ax3[0, 0].imshow(ref_recon, vmax=vmax, cmap=plt.cm.Greys)
ax3[0, 0].set_title('REFERENCE', fontsize='small')
ax3[0, 1].imshow(recon_osem, vmax=vmax, cmap=plt.cm.Greys)
ax3[0, 1].set_title('OSEM', fontsize='small')
ax3[1, 0].imshow(gaussian_filter(ref_recon, sig), vmax=vmax, cmap=plt.cm.Greys)
ax3[1, 0].set_title('ps REFERNCE', fontsize='small')
ax3[1, 1].imshow(gaussian_filter(recon_osem, sig),
                 vmax=vmax,
                 cmap=plt.cm.Greys)
ax3[1, 1].set_title('ps OSEM', fontsize='small')

for ig, gamma in enumerate(gammas):
    ax3[0, ig + 2].imshow(recons_spdhg[ig, ...], vmax=vmax, cmap=plt.cm.Greys)
    ax3[0, ig + 2].set_title(f'SPDHGw {gamma:.1E}', fontsize='small')
    ax3[1, ig + 2].imshow(gaussian_filter(recons_spdhg[ig, ...], sig),
                          vmax=vmax,
                          cmap=plt.cm.Greys)
    ax3[1, ig + 2].set_title(f'ps SPDHGw {gamma:.1E}', fontsize='small')

    ax3[2, ig + 2].imshow(recons_spdhg2[ig, ...], vmax=vmax, cmap=plt.cm.Greys)
    ax3[2, ig + 2].set_title(f'SPDHGc {gamma:.1E}', fontsize='small')
    ax3[3, ig + 2].imshow(gaussian_filter(recons_spdhg2[ig, ...], sig),
                          vmax=vmax,
                          cmap=plt.cm.Greys)
    ax3[3, ig + 2].set_title(f'ps SPDHGc {gamma:.1E}', fontsize='small')

for axx in ax3.flatten():
    axx.set_axis_off()

fig3.tight_layout()
fig3.savefig(os.path.join('figs', f'{base_str}.png'))
fig3.show()

# show the early reconstructions
fig4, ax4 = plt.subplots(4,
                         len(gammas) + 2,
                         figsize=((len(gammas) + 2) * 2, 8),
                         squeeze=False)

vmax = 1.5 * img.max()

ax4[0, 0].imshow(ref_recon, vmax=vmax, cmap=plt.cm.Greys)
ax4[0, 0].set_title('REFERENCE', fontsize='small')
ax4[0, 1].imshow(recon_osem_early, vmax=vmax, cmap=plt.cm.Greys)
ax4[0, 1].set_title('OSEM', fontsize='small')
ax4[1, 0].imshow(gaussian_filter(ref_recon, sig), vmax=vmax, cmap=plt.cm.Greys)
ax4[1, 0].set_title('ps REFERNCE', fontsize='small')
ax4[1, 1].imshow(gaussian_filter(recon_osem_early, sig),
                 vmax=vmax,
                 cmap=plt.cm.Greys)
ax4[1, 1].set_title('ps OSEM', fontsize='small')

for ig, gamma in enumerate(gammas):
    ax4[0, ig + 2].imshow(recons_spdhg_early[ig, ...],
                          vmax=vmax,
                          cmap=plt.cm.Greys)
    ax4[0, ig + 2].set_title(f'SPDHGw {gamma:.1E}', fontsize='small')
    ax4[1, ig + 2].imshow(gaussian_filter(recons_spdhg_early[ig, ...], sig),
                          vmax=vmax,
                          cmap=plt.cm.Greys)
    ax4[1, ig + 2].set_title(f'ps SPDHGw {gamma:.1E}', fontsize='small')

    ax4[2, ig + 2].imshow(recons_spdhg_early2[ig, ...],
                          vmax=vmax,
                          cmap=plt.cm.Greys)
    ax4[2, ig + 2].set_title(f'SPDHGc {gamma:.1E}', fontsize='small')
    ax4[3, ig + 2].imshow(gaussian_filter(recons_spdhg_early2[ig, ...], sig),
                          vmax=vmax,
                          cmap=plt.cm.Greys)
    ax4[3, ig + 2].set_title(f'ps SPDHGc {gamma:.1E}', fontsize='small')

for axx in ax4.flatten():
    axx.set_axis_off()

fig4.tight_layout()
fig4.savefig(os.path.join('figs', f'{base_str}_early_{it_early}.png'))
fig4.show()
