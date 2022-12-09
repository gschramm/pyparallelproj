"""OSEM reconstruction example using simulated brainweb data"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--symmetry_axis', type=int, default=1, choices=[0, 1, 2])
args = parser.parse_args()

mode = 'CPU'

if mode == 'GPU':
    import cupy as cp
    xp = cp
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif mode == 'hybrid':
    xp = np
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif mode == 'CPU':
    xp = np
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    raise ValueError

import pyparallelproj.coincidences as coincidences
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors

# image properties
num_trans = 215
num_ax = 71
voxel_size = (2.78, 2.78, 2.78)

# scanner properties
num_rings = 36
symmetry_axis = args.symmetry_axis
fwhm_mm_recon = 4.0
tof_parameters = tof.ge_discovery_mi_tof_parameters

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

tofbin = 0
projector.events = xp.array([[19, 466, 15, 235, tofbin]], dtype=xp.int16)
print(projector.forward_listmode(xp.ones(image_shape, dtype=xp.float32)))