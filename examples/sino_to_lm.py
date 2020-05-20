import sys
import os

if not os.path.abspath('..') in sys.path: sys.path.append(os.path.abspath('..'))

import pyparallelproj as ppp
import numpy as np

#---------------------------------------------------------------------------------

ngpus       = 0
nsubsets    = 1
subset      = 0 

np.random.seed(1)

# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,3]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2])


# setup a random image
img = np.zeros((n0,n1,n2))
img[(n0//8):(7*n0//8),(n1//8):(7*n1//8),:] = 0.04
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

######## tof projection
tofsino_params = ppp.PETSinogram(scanner, ntofbins = 27, tofbin_width = 28.)
tofproj        = ppp.SinogramProjector(scanner, tofsino_params, img.shape, nsubsets = nsubsets, 
                                       voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                       tof = True, sigma_tof = 60., n_sigmas = 3)

img_fwd_tof = tofproj.fwd_project(img, subset = subset)


# generate a noise realization
noisy_sino_tof = np.random.poisson(img_fwd_tof)

# events is a list of all events
# each event if characterize by 5 integers: 
# [start_crystal_id_tr, start_crystal_id_ax, end_crystal_id_tr, end_crystal_id_ax, tofbin]

events = []

it = np.nditer(noisy_sino_tof, flags=['multi_index'])
for x in it:
  if x > 0:
    event      = np.zeros(5, dtype = np.int16)
    event[0:2] = tofproj.istart[it.multi_index[:-1]]
    event[2:4] = tofproj.iend[it.multi_index[:-1]]
    event[4]   = it.multi_index[-1]

    t = int(x)*[event]
    events += t

events = np.array(events)

# shuffle the event order
tmp = np.arange(events.shape[0])
np.random.shuffle(tmp)
events = events[tmp,:]
