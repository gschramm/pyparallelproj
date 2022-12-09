"""OSEM reconstruction example using simulated brainweb data"""
from time import time
import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import json
from pathlib import Path
import h5py

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--num_events', type=int, default=40000000)
parser.add_argument('--num_iterations', type=int, default=4)
parser.add_argument('--num_subsets', type=int, default=34)
parser.add_argument('--mode', default='GPU', choices=['GPU', 'CPU', 'hybrid'])
parser.add_argument('--threadsperblock', type=int, default=32)
parser.add_argument('--output_file', type=int, default=None)
parser.add_argument('--output_dir', default='results')
parser.add_argument('--presort', action='store_true')
parser.add_argument('--post_sm_fwhm', type=float, default=6.)
parser.add_argument('--symmetry_axis', type=int, default=2, choices=[0, 1, 2])
args = parser.parse_args()

presort = args.presort
post_sm_fwhm = args.post_sm_fwhm

if args.mode == 'GPU':
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    xp = cp
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.mode == 'hybrid':
    import scipy.ndimage as ndi
    xp = np
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.mode == 'CPU':
    import scipy.ndimage as ndi
    xp = np
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    raise ValueError

import pyparallelproj.coincidences as coincidences
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.listmode_algorithms as lm_algorithms

data_str = 'tof_listmode'
if presort:
    data_str += '_presorted'

#output_dir = args.output_dir
#if args.output_file is None:
#    output_file = f'{data_str}__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numevents_{num_events}.csv'

# image properties
num_trans = 215
num_ax = 71
voxel_size = (2.78, 2.78, 2.78)

# scanner properties
num_rings = 36
symmetry_axis = args.symmetry_axis
fwhm_mm_recon = 4.0
tof_parameters = tof.ge_discovery_mi_tof_parameters

# reconstruction parameters
num_iterations = args.num_iterations
num_subsets = args.num_subsets
threadsperblock = args.threadsperblock
num_events = args.num_events
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- define the scanner geometry -------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

image_shape = 3 * [num_trans]
image_shape[symmetry_axis] = num_ax
image_shape = tuple(image_shape)

image_origin = tuple(
    (-0.5 * image_shape[i] + 0.5) * voxel_size[i] for i in range(3))

coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
    num_rings=36, symmetry_axis=symmetry_axis, xp=xp)
#---------------------------------------------------------------------
#--- setup of the PET forward model (the projector) ------------------
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# setup a non-time-of-flight and time-of-flight projector
# to calculate the attenuation factors based on the attenuation image
projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                             image_shape, image_origin,
                                             voxel_size)

projector.tof_parameters = tof.ge_discovery_mi_tof_parameters

for i in np.arange(-13, 14):
    projector.events = xp.array([[19, 466, 15, 235, i]], dtype=xp.int16)
    print(projector.forward_listmode(xp.ones(image_shape, dtype=xp.float32)))