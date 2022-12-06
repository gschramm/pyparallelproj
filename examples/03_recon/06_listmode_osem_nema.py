"""OSEM reconstruction example using simulated brainweb data"""

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#
# make sure to run the script "download_brainweb_petmr.py" in ../data
# before running this script
#
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

import json
from time import time
import h5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.listmode_algorithms as lm_algorithms

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

# variable that determines whether to use cupy (cp) or numpy (np) array for computations
# if cupy (cp), there is no memory transfer between host and GPU
xp = cp

if xp.__name__ == 'cupy':
    import cupyx.scipy.ndimage as ndi
else:
    import scipy.ndimage as ndi

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- input parmeters -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#-------------------
# reconstruction parameters
num_iterations = 4
num_subsets = 34
num_events = 40000000

#-------------------
# scanner parameters
symmetry_axis = 2
fwhm_mm_recon = 4.0

#-------------------
# sinogram (data order) parameters
sinogram_order = 'RVP'

#-------------------
# image parameters
voxel_size = (2.78, 2.78, 2.78)
image_shape = (135, 135, 71)
#image_shape = (215, 215, 71)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- define the scanner geometry -------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

image_origin = tuple(
    (-0.5 * image_shape[i] + 0.5) * voxel_size[i] for i in range(3))

scanner = scanners.GEDiscoveryMI(36, symmetry_axis=symmetry_axis, xp=xp)

# setup the coincidence descriptor
coincidence_descriptor = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=65,
    max_ring_difference=scanner.num_rings - 1,
    sinogram_spatial_axis_order=coincidences.
    SinogramSpatialAxisOrder[sinogram_order])

#---------------------------------------------------------------------
#--- setup of the PET forward model (the projector) ------------------
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# setup a non-time-of-flight and time-of-flight projector
# to calculate the attenuation factors based on the attenuation image
projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                             image_shape, image_origin,
                                             voxel_size)

#--------------------------------------------------------------------------
# use an image-based resolution model in the projector to model the effect
# of limited resolution
res_model = resolution_models.GaussianImageBasedResolutionModel(
    image_shape, tuple(fwhm_mm_recon / (2.35 * x) for x in voxel_size), xp,
    ndi)

projector.image_based_resolution_model = res_model

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- load sensitivity / attenuation "singorams" in LM-----------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# approximate the adjoint of ones, but calculating a non-TOF listmode
# backprojection over all crystal pairs

with open('.nema_data.json', 'r') as jfile:
    data_path = Path(json.load(jfile)['path'])

with h5py.File(data_path / 'corrections.h5', 'r') as data:
    all_multiplicative_factors = data['all_xtals/atten'][:] * data[
        'all_xtals/sens'][:]
    all_xtals = data['all_xtals/xtal_ids'][:][:, [1, 0, 3, 2]]

if xp.__name__ == 'cupy':
    print('copying data to GPU')
    all_xtals = xp.asarray(all_xtals)
    all_multiplicative_factors = xp.asarray(all_multiplicative_factors)

projector.events = all_xtals
projector.multiplicative_correction_list = all_multiplicative_factors

t0 = time()
adjoint_ones = projector.adjoint_listmode(
    xp.ones(all_xtals.shape[0], dtype=xp.float32))
t1 = time()

print(f'time to calculate non-tof adjoint ones {(t1-t0):.2F}s')

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- load listmode data ----------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

with h5py.File(data_path / 'LIST0000.BLF', 'r') as data:
    events = data['MiceList/TofCoinc'][:]

if num_events is None:
    num_events = events.shape[0]

with h5py.File(data_path / 'corrections.h5', 'r') as data:
    multiplicative_correction_list = data['correction_lists/sens'][:] * data[
        'correction_lists/atten'][:]
    contamination_list = data['correction_lists/contam'][:]

# shuffle events since events come semi sorted
print('shuffling LM data')
num_all_events = events.shape[0]
ie = np.arange(num_all_events)
np.random.shuffle(ie)
events = events[ie, :]
multiplicative_correction_list = multiplicative_correction_list[ie]
contamination_list = contamination_list[ie]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed
events[:, -1] = -(events[:, -1] // 13)

## use only part of the events
if num_events is not None:
    multiplicative_correction_list = multiplicative_correction_list[:
                                                                    num_events]
    contamination_list = contamination_list[:num_events] * (num_events /
                                                            num_all_events)
    events = events[:num_events, :]

if xp.__name__ == 'cupy':
    print('copying data to GPU')
    events = xp.asarray(events)
    multiplicative_correction_list = xp.asarray(multiplicative_correction_list)
    contamination_list = xp.asarray(contamination_list)

projector.events = events
projector.multiplicative_correction_list = multiplicative_correction_list

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- run LM OSEM reconstruction --------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# set tof parameters of projector to enable TOF projections
speed_of_light = 300.  # [mm/ns]
time_res_FWHM = 0.385  # [ns]

tof_parameters = tof.TOFParameters(
    num_tofbins=29,
    tofbin_width=13 * 0.01302 * speed_of_light / 2,
    sigma_tof=(speed_of_light / 2) * (time_res_FWHM / 2.355),
    num_sigmas=3)

projector.tof_parameters = tof_parameters
projector.listmode_subsetter.num_subsets = num_subsets

print('starting reconstruction')
listmode_reconstructor = lm_algorithms.LM_OSEM(contamination_list, projector,
                                               adjoint_ones)
listmode_reconstructor.run(num_iterations)

x_lm = listmode_reconstructor.x

print(f'time per iteration {np.diff(listmode_reconstructor.walltime)}s')

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---- visualizations -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# get arrays from GPU if running with cupy
if xp.__name__ == 'cupy':
    x_lm = xp.asnumpy(x_lm)

x_lm_sm = gaussian_filter(x_lm, 6. / (2.35 * np.array(voxel_size)))

fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex='col', sharey='col')
ims = dict(cmap=plt.cm.Greys,
           origin='lower',
           vmin=0,
           vmax=0.35 * num_events / 4e7)
ax[0, 0].imshow(x_lm[:, :, 51].T, **ims)
ax[0, 1].imshow(x_lm[:, image_shape[1] // 2, :].T, **ims)
ax[1, 0].imshow(x_lm_sm[:, :, 51].T, **ims)
ax[1, 1].imshow(x_lm_sm[:, image_shape[1] // 2, :].T, **ims)
fig.tight_layout()
fig.show()