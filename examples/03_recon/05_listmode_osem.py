"""OSEM reconstruction example using simulated brainweb data"""

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#
# make sure to run the script "download_brainweb_petmr.py" in ../data
# before running this script
#
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.algorithms as algorithms
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
num_iterations = 6
num_subsets = 34

#-------------------
# scanner parameters
num_rings = 1
symmetry_axis = 2
fwhm_mm_data = 4.5
fwhm_mm_recon = 4.5

# global sensitivity factor of the scanner that can be used to change
# the norm of the forward operator
scanner_sensitivty = 1. / 30

#-------------------
# sinogram (data order) parameters
sinogram_order = 'RVP'

#-------------------
# image parameters
voxel_size = (2., 2., 2.)

# brainweb subset number
# [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
subject_number = 38
# simulation number [0,1,2]
sim_number = 0

# number of true emitted coincidences per volume (mm^3)
# 5 -> low counts, 5 -> medium counts, 500 -> high counts
trues_per_volume = 50.

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- setup the emission and attenuation images -----------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

nii = nib.as_closest_canonical(
    nib.load(
        f'../data/brainweb_petmr/subject{subject_number:02}/sim_{sim_number}/true_pet.nii.gz'
    ))
image = xp.array(nii.get_fdata(), dtype=xp.float32)

# downsample image by a factor of 2, to get 2mm voxels
image = (image[::2, :, :] + image[1::2, :, :]) / 2
image = (image[:, ::2, :] + image[:, 1::2, :]) / 2
image = (image[:, :, ::2] + image[:, :, 1::2]) / 2

num_axial = max(int(5.5 * num_rings / voxel_size[symmetry_axis]), 1)
if num_rings == 1:
    num_axial = 1

start_sl = image.shape[2] // 2 - num_axial // 2
end_sl = start_sl + num_axial
image = image[:, :, start_sl:end_sl]

image_shape = image.shape

image_origin = tuple(
    (-0.5 * image_shape[i] + 0.5) * voxel_size[i] for i in range(3))

attenuation_image = (0.01 * (image > 0)).astype(xp.float32)

#---------------------------------------------------------------------
#--- setup of the PET forward model (the projector) ------------------
#---------------------------------------------------------------------

coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
    num_rings=num_rings,
    sinogram_spatial_axis_order=coincidences.
    SinogramSpatialAxisOrder[sinogram_order],
    xp=xp)

#---------------------------------------------------------------------
# setup a non-time-of-flight and time-of-flight projector
# to calculate the attenuation factors based on the attenuation image
projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                             image_shape, image_origin,
                                             voxel_size)

# simulate the attenuation factors (exp(-fwd(attenuation_image)))
attenuation_factors = xp.exp(-projector.forward(attenuation_image))

projector.tof_parameters = tof.ge_discovery_mi_tof_parameters

#--------------------------------------------------------------------------
# use an image-based resolution model in the projector to model the effect
# of limited resolution
res_model = resolution_models.GaussianImageBasedResolutionModel(
    image_shape, tuple(fwhm_mm_data / (2.35 * x) for x in voxel_size), xp, ndi)

projector.image_based_resolution_model = res_model

#--------------------------------------------------------------------------
# set the multiplicative corrections in the scanner to model photon attenuation
# and scanner sensitivity

projector.multiplicative_corrections = attenuation_factors * scanner_sensitivty

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--- simulate acquired data based on forward model and known contaminations ---
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

image_fwd = projector.forward(image)

# scale the image such that we get a certain true count per emission voxel value
emission_volume = xp.where(image > 0)[0].shape[0] * np.prod(voxel_size)
current_trues_per_volume = float(image_fwd.sum() / emission_volume)

scaling_factor = (trues_per_volume / current_trues_per_volume)
image *= scaling_factor
image_fwd *= scaling_factor

# simulate a constant background contamination
contamination = xp.full(projector.output_shape,
                        image_fwd.mean(),
                        dtype=xp.float32)

# generate noisy data
data = xp.random.poisson(image_fwd + contamination).astype(xp.uint16)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- generate listmode data ------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

non_zero_bins = xp.where(data > 0)
non_zero_data = data[non_zero_bins]

lor_nums = np.repeat(xp.asnumpy(non_zero_bins[0]), xp.asnumpy(non_zero_data))
tof_bins = np.repeat(
    xp.asnumpy(non_zero_bins[1]),
    xp.asnumpy(non_zero_data)) - projector.tof_parameters.num_tofbins // 2

# get the detecor numbers from the LOR numbers of the events
start_mod, start_ind, end_mod, end_ind = coincidence_descriptor.get_lor_indices(
    lor_nums)

events = xp.array([start_mod, start_ind, end_mod, end_ind, tof_bins]).T

contamination_list = contamination[lor_nums, tof_bins]
multiplicative_correction_list = attenuation_factors[
    lor_nums] * scanner_sensitivty

## shuffle the events
event_index = np.arange(events.shape[0])
np.random.shuffle(event_index)

events = events[event_index, :]
contamination_list = contamination_list[event_index]
multiplicative_correction_list = multiplicative_correction_list[event_index]

projector.events = events
projector.multiplicative_correction_list = multiplicative_correction_list

adjoint_ones = projector.adjoint(
    xp.ones(projector.output_shape, dtype=xp.float32))

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- run LM OSEM reconstruction --------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

projector.listmode_subsetter.num_subsets = num_subsets

listmode_reconstructor = lm_algorithms.LM_OSEM(contamination_list, projector,
                                               adjoint_ones)
listmode_reconstructor.run(num_iterations)

x_lm = listmode_reconstructor.x

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- run sinogram OSEM reconstruction --------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# add a subsetter to the projector such that we can do updates with ordered subsets
subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor, num_subsets)
projector.subsetter = subsetter

reconstructor = algorithms.OSEM(data, contamination, projector, verbose=True)
reconstructor.run(num_iterations, evaluate_cost=False)

x = reconstructor.x

print(
    f'final cost OSEM    ..: {reconstructor.evaluate_cost(reconstructor.x):.5E}'
)
print(
    f'final cost LM-OSEM ..: {reconstructor.evaluate_cost(listmode_reconstructor.x):.5E}'
)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---- visualizations -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# get arrays from GPU if running with cupy
if xp.__name__ == 'cupy':
    data = xp.asnumpy(data)
    x = xp.asnumpy(x)
    x_lm = xp.asnumpy(x_lm)
    image = xp.asnumpy(image)

# reshape the data into a sinogram (just for visualizations)
data_reshaped = data.reshape(coincidence_descriptor.sinogram_spatial_shape +
                             (projector.tof_parameters.num_tofbins, ))

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ims = dict(cmap=plt.cm.Greys, vmin=0, vmax=1.2 * image.max(), origin='lower')
ax[0].imshow(image[:, :, image_shape[2] // 2].T, **ims)
ax[1].imshow(x[:, :, image_shape[2] // 2].T, **ims)
ax[2].imshow(x_lm[:, :, image_shape[2] // 2].T, **ims)
ax[3].imshow(data_reshaped[:, :, num_rings // 2,
                           projector.tof_parameters.num_tofbins // 2].T,
             cmap=plt.cm.Greys)
fig.tight_layout()
fig.show()