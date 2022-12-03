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
import pyparallelproj.acquisition_models as acquisition_models
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.algorithms as algorithms

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
# input parmeters

# scanner parameters
num_rings = 1
symmetry_axis = 2

fwhm_mm_data = 4.5
fwhm_mm_recon = 4.5

voxel_size = (2., 2., 2.)

#-------------------
# image parameters

# brainweb subset number
# [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
subject_number = 38
# simulation number [0,1,2]
sim_number = 0

#-------------------
# sinogram (data order) parameters
sinogram_order = 'RVP'

#-------------------
# reconstruction parameters
num_iterations = 6
num_subsets = 28

# number of true emitted coincidences per volume (mm^3)
# 5 -> low counts, 5 -> medium counts, 500 -> high counts
trues_per_volume = 50.

# global sensitivity factor of the scanner that can be used to
scanner_sensitivty = 1.
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# define the scanner geometry
scanner = scanners.GEDiscoveryMI(num_rings, symmetry_axis=symmetry_axis, xp=xp)

# setup the coincidence descriptor
coincidence_descriptor = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=65,
    max_ring_difference=scanner.num_rings - 1,
    sinogram_spatial_axis_order=coincidences.
    SinogramSpatialAxisOrder[sinogram_order])

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

num_axial = max(
    int((scanner.all_lor_endpoints[:, symmetry_axis].max() -
         scanner.all_lor_endpoints[:, symmetry_axis].min()) /
        voxel_size[symmetry_axis]), 1)

start_sl = image.shape[2] // 2 - num_axial // 2
end_sl = start_sl + num_axial
image = image[:, :, start_sl:end_sl]

image_shape = image.shape

image_origin = tuple(
    (-0.5 * image_shape[i] + 0.5) * voxel_size[i] for i in range(3))

attenuation_image = (0.01 * (image > 0)).astype(xp.float32)

#---------------------------------------------------------------------

subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor, num_subsets)

# setup a non-time-of-flight and time-of-flight projector
nontof_projector = petprojectors.NonTOFPETJosephProjector(
    coincidence_descriptor, image_shape, image_origin, voxel_size, subsetter)

# tof parameters
speed_of_light = 300.  # [mm/ns]
time_res_FWHM = 0.385  # [ns]

tof_parameters = tof.TOFParameters(
    num_tofbins=29,
    tofbin_width=13 * 0.01302 * speed_of_light / 2,
    sigma_tof=(speed_of_light / 2) * (time_res_FWHM / 2.355),
    num_sigmas=3)

projector = petprojectors.TOFPETJosephProjector(coincidence_descriptor,
                                                image_shape, image_origin,
                                                voxel_size, subsetter,
                                                tof_parameters)

# simulate the attenuation factors (exp(-fwd(attenuation_image)))
attenuation_factors = xp.exp(-nontof_projector.forward(attenuation_image))

# simulate LOR sensitivity factors
sensitivity_factors = xp.full(nontof_projector.output_shape,
                              scanner_sensitivty,
                              dtype=xp.float32)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape, 1e-3, dtype=xp.float32)

res_model_data = resolution_models.GaussianImageBasedResolutionModel(
    image_shape, tuple(fwhm_mm_data / (2.35 * x) for x in voxel_size), xp, ndi)

# setup the forward operator ("A") that also supports subsets
acq_model_data = acquisition_models.PETAcquisitionModel(
    projector,
    attenuation_factors,
    sensitivity_factors,
    image_based_resolution_model=res_model_data)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# simulate acquired data based on forward model and known contaminations

tmp = acq_model_data.forward(image)

# scale the image such that we get a certain true count per emission voxel value
emission_volume = xp.where(image > 0)[0].shape[0] * np.prod(voxel_size)
current_trues_per_volume = float(tmp.sum() / emission_volume)
image *= (trues_per_volume / current_trues_per_volume)
del tmp
image_fwd = acq_model_data.forward(image)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape,
                        image_fwd.mean(),
                        dtype=xp.float32)

# generate noisy data
data = xp.random.poisson(image_fwd + contamination).astype(xp.uint16)

del acq_model_data

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# run an OSEM reconstruction

res_model_recon = resolution_models.GaussianImageBasedResolutionModel(
    image_shape, tuple(fwhm_mm_recon / (2.35 * x) for x in voxel_size), xp,
    ndi)

# setup the forward operator ("A") that also supports subsets
acq_model_recon = acquisition_models.PETAcquisitionModel(
    projector,
    attenuation_factors,
    sensitivity_factors,
    image_based_resolution_model=res_model_recon)

reconstructor = algorithms.OSEM(data,
                                contamination,
                                acq_model_recon,
                                verbose=True)
reconstructor.run(num_iterations, evaluate_cost=False)

x = reconstructor.x

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# visualizations

# get arrays from GPU if running with cupy
if xp.__name__ == 'cupy':
    data = xp.asnumpy(data)
    x = xp.asnumpy(x)
    image = xp.asnumpy(image)

# reshape the data into a sinogram (just for visualizations)
data_reshaped = data.reshape(coincidence_descriptor.sinogram_spatial_shape +
                             (tof_parameters.num_tofbins, ))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ims = dict(cmap=plt.cm.Greys, vmin=0, vmax=1.2 * image.max(), origin='lower')
ax[0].imshow(image[:, :, image_shape[2] // 2].T, **ims)
ax[1].imshow(x[:, :, image_shape[2] // 2].T, **ims)
ax[2].imshow(data_reshaped[:, :, num_rings // 2,
                           tof_parameters.num_tofbins // 2].T,
             cmap=plt.cm.Greys)
fig.tight_layout()
fig.show()