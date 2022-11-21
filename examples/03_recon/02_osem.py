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
radius = 350
num_sides = 28
num_lor_endpoints_per_side = 16
lor_spacing = 4.
# number of detector rings, 1: single ring -> 2D example, 27: "short 3D scanner"
num_rings = 1

max_ring_difference = num_rings - 1
radial_trim = 149

ring_positions = 5.5 * (np.arange(num_rings) - num_rings / 2 + 0.5)
symmetry_axis = 2

fwhm_mm_data = 4.5
fwhm_mm_recon = 4.5

#-------------------
# image parameters

# brainweb subset number
# [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
subsetject_number = 38
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

voxsize = (2., 2., 2.)

nii = nib.as_closest_canonical(
    nib.load(
        f'../data/brainweb_petmr/subject{subsetject_number:02}/sim_{sim_number}/true_pet.nii.gz'
    ))
img = xp.array(nii.get_fdata(), dtype=xp.float32)

# downsample image by a factor of 2, to get 2mm voxels
img = (img[::2, :, :] + img[1::2, :, :]) / 2
img = (img[:, ::2, :] + img[:, 1::2, :]) / 2
img = (img[:, :, ::2] + img[:, :, 1::2]) / 2

num_axial = max(
    int((ring_positions.max() - ring_positions.min()) /
        voxsize[symmetry_axis]), 1)

start_sl = img.shape[2] // 2 - num_axial // 2
end_sl = start_sl + num_axial
img = img[:, :, start_sl:end_sl]

img_shape = img.shape

img_origin = tuple((-0.5 * img_shape[i] + 0.5) * voxsize[i] for i in range(3))

attenuation_img = (0.01 * (img > 0)).astype(xp.float32)

#---------------------------------------------------------------------

scanner = scanners.RegularPolygonPETScannerGeometry(
    radius,
    num_sides,
    num_lor_endpoints_per_side,
    lor_spacing,
    num_rings,
    ring_positions,
    symmetry_axis=symmetry_axis,
    xp=xp)

# setup the coincidence descriptor
coincidence_descriptor = coincidences.RegularPolygonPETCoincidenceDescriptor(
    scanner,
    radial_trim=radial_trim,
    max_ring_difference=max_ring_difference,
    sinogram_spatial_axis_order=coincidences.
    SinogramSpatialAxisOrder[sinogram_order])

subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor, num_subsets)

# setup a non-time-of-flight and time-of-flight projector
nontof_projector = petprojectors.NonTOFPETJosephProjector(
    coincidence_descriptor, img_shape, img_origin, voxsize, subsetter)

tof_parameters = tof.TOFParameters(num_tofbins=27,
                                   tofbin_width=22.2,
                                   sigma_tof=60 / 2.35)

projector = petprojectors.TOFPETJosephProjector(coincidence_descriptor,
                                                img_shape, img_origin, voxsize,
                                                subsetter, tof_parameters)

# simulate the attenuation factors (exp(-fwd(attenuation_image)))
attenuation_factors = xp.exp(-nontof_projector.forward(attenuation_img))

# simulate LOR sensitivity factors
sensitivity_factors = xp.full(nontof_projector.output_shape,
                              scanner_sensitivty,
                              dtype=xp.float32)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape, 1e-3, dtype=xp.float32)

res_model_data = resolution_models.GaussianImageBasedResolutionModel(
    img_shape, tuple(fwhm_mm_data / (2.35 * x) for x in voxsize), xp, ndi)

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

tmp = acq_model_data.forward(img)

# scale the image such that we get a certain true count per emission voxel value
emission_volume = xp.where(img > 0)[0].shape[0] * np.prod(voxsize)
current_trues_per_volume = float(tmp.sum() / emission_volume)
img *= (trues_per_volume / current_trues_per_volume)
del tmp
img_fwd = acq_model_data.forward(img)

# simulate a constant background contamination
contamination = xp.full(projector.output_shape,
                        img_fwd.mean() / 10.,
                        dtype=xp.float32)

# generate noisy data
data = xp.random.poisson(img_fwd + contamination).astype(xp.uint16)

del acq_model_data

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# run an OSEM reconstruction

res_model_recon = resolution_models.GaussianImageBasedResolutionModel(
    img_shape, tuple(fwhm_mm_recon / (2.35 * x) for x in voxsize), xp, ndi)

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
    img = xp.asnumpy(img)

# reshape the data into a sinogram (just for visualizations)
data_reshaped = data.reshape(coincidence_descriptor.sinogram_spatial_shape +
                             (tof_parameters.num_tofbins, ))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ims = dict(cmap=plt.cm.Greys, vmin=0, vmax=1.2 * img.max(), origin='lower')
ax[0].imshow(img[:, :, img_shape[2] // 2].T, **ims)
ax[1].imshow(x[:, :, img_shape[2] // 2].T, **ims)
ax[2].imshow(data_reshaped[:, :, num_rings // 2,
                           tof_parameters.num_tofbins // 2].T,
             cmap=plt.cm.Greys)
fig.tight_layout()
fig.show()