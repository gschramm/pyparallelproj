import sys
import os

if not os.path.abspath('..') in sys.path: sys.path.append(os.path.abspath('..'))

import pyparallelproj as ppp
import numpy as np

#---------------------------------------------------------------------------------

ngpus       = 0
nsubsets    = 3
subset      = 2 

# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = 1


# setup a random image
img = np.random.rand(n0,n1,n2)
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

######## nontof projections
sino_params = ppp.PETSinogram(scanner)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus)

# setup a random sinogram
rsino = np.random.rand(*proj.subset_sino_shapes[0])

img_fwd = proj.fwd_project(img, subset = subset)
back    = proj.back_project(rsino, subset = subset)

######## tof projections
tofsino_params = ppp.PETSinogram(scanner, ntofbins = 27, tofbin_width = 28.)
tofproj        = ppp.SinogramProjector(scanner, tofsino_params, img.shape, nsubsets = nsubsets, 
                                       voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                       tof = True, sigma_tof = 60., n_sigmas = 3)

# setup a random sinogram
tsino = np.random.rand(*tofproj.subset_sino_shapes[0])

img_fwd_tof = tofproj.fwd_project(img, subset = subset)
back_tof    = proj.back_project(tsino, subset = subset)