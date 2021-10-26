# script to investigate the convergence of SPDHG for 2D TOF PET

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.algorithms import osem, spdhg
from pyparallelproj.phantoms   import ellipse2d_phantom, brain2d_phantom

import numpy as np
from scipy.ndimage import gaussian_filter

from pyparallelproj.utils import GradientOperator, GradientNorm

import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 10,  type = int)
parser.add_argument('--nsubsets',   help = 'number of subsets',     default = 56,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',  help = 'prior strength',  default = 0.1, type = float)
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed
beta          = args.beta

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
  n   = 200
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = ellipse_phantom(n = n, c = 3)
  for i2 in range(n2):
    img[:,:,i2] = tmp
elif phantom == 'brain2d':
  n   = 128
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = brain2d_phantom(n = n)
  for i2 in range(n2):
    img[:,:,i2] = tmp

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)


# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype = np.float32)

# estimate the norm of the operator
test_img = np.random.rand(*img.shape)
for i in range(10):
  fwd  = ppp.pet_fwd_model(test_img, proj, attn_sino, sens_sino, fwhm = fwhm)
  back = ppp.pet_back_model(fwd, proj, attn_sino, sens_sino, fwhm = fwhm)

  pnsq = np.linalg.norm(back)
  print(i,np.sqrt(pnsq))

  test_img = back / pnsq

# normalize sensitivity sinogram to get PET forward model for 1 view with norm 1
# this is important otherwise the step size T in SPDHG get dominated by the gradient
sens_sino /= (np.sqrt(pnsq)/np.sqrt(8))
# forward project the image
img_fwd= ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

# setup joint gradient field
G0            = GradientOperator()
grad_operator = GradientOperator(joint_grad_field = G0.fwd(-(img**0.5)))
grad_norm     = GradientNorm(beta = beta)

recon = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
              gamma = 1/img.max(), fwhm = fwhm, verbose = True, 
              xstart = None, grad_operator = grad_operator, grad_norm = grad_norm)

fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].imshow(img.squeeze(), cmap = plt.cm.Greys, vmin = 0, vmax = 1.2*img.max())
ax[1].imshow(recon.squeeze(), cmap = plt.cm.Greys, vmin = 0, vmax = 1.2*img.max())
fig.tight_layout()
fig.show()