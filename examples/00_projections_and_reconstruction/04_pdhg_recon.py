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

import pyparallelproj.coincidences as coincidences
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.algorithms as algorithms
import pyparallelproj.operators as operators
import pyparallelproj.functionals as functionals

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
num_iterations = 500

# prior parameters
beta = 5e-3
prior_norm = functionals.L2L1Norm(xp)
#beta = 3e-4
#prior_norm = functionals.SquaredL2Norm(xp)

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
#--- run OSEM reconstruction -----------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# update the resolution model which can be different for the reconstruction
res_model_recon = resolution_models.GaussianImageBasedResolutionModel(
    image_shape, tuple(fwhm_mm_recon / (2.35 * x) for x in voxel_size), xp,
    ndi)

projector.image_based_resolution_model = res_model_recon

data_distance = functionals.NegativePoissonLogLikelihood(data, xp)
prior_operator = operators.GradientOperator(projector.input_shape, xp)

projector_norm = projector.norm()
gam = 1 / image.max()
sigma = gam * 0.9 / float(projector_norm)
tau = 0.9 / float(projector_norm) / gam

reconstructor = algorithms.PDHG(
    projector,
    data_distance,
    prior_operator,
    prior_norm,
    beta,
    sigma,
    tau,
    contamination=contamination,
    g_functional=functionals.BoundIndicatorFunctional(xp, lb=0., ub=xp.inf))

reconstructor.run(num_iterations, calculate_cost=True)

x = reconstructor.x

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---- visualizations -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# get arrays from GPU if running with cupy
if xp.__name__ == 'cupy':
    data = xp.asnumpy(data)
    x = xp.asnumpy(x)
    image = xp.asnumpy(image)

# reshape the data into a sinogram (just for visualizations)
data_reshaped = data.reshape(coincidence_descriptor.sinogram_spatial_shape +
                             (projector.tof_parameters.num_tofbins, ))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ims = dict(cmap=plt.cm.Greys, vmin=0, vmax=1.2 * image.max(), origin='lower')
ax[0].imshow(image[:, :, image_shape[2] // 2].T, **ims)
ax[1].imshow(x[:, :, image_shape[2] // 2].T, **ims)
ax[2].imshow(data_reshaped[:, :, num_rings // 2,
                           projector.tof_parameters.num_tofbins // 2].T,
             cmap=plt.cm.Greys)
fig.tight_layout()
fig.show()