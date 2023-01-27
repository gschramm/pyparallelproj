"""script to generate 2D PET data based on the brainweb phantom"""

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#
# make sure to run the script "download_brainweb_petmr.py" in ../data
# before running this script
#
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import dill

import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.algorithms as algorithms

import cupy as cp
import cupyx.scipy.ndimage as ndi

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- input parmeters -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# seed for the random generator
seed = 1
cp.random.seed(seed)

#-------------------
# reconstruction parameters
num_iterations = 4
num_subsets = 34

#-------------------
# scanner parameters
num_rings = 1
symmetry_axis = 0
fwhm_mm_data = 4.5
fwhm_mm_recon = 4.5

#-------------------
# sinogram (data order) parameters
sinogram_order = 'RVP'

#-------------------
# image parameters
voxel_size = (2., 2., 2.)

# number of true emitted coincidences per volume (mm^3)
# 5 -> low counts, 5 -> medium counts, 500 -> high counts
trues_per_volume = 50.

for sim_number in [0, 1, 2]:
    for subject_number in [
            4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            52, 53, 54
    ]:
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #--- setup the emission and attenuation images -----------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------

        nii = nib.as_closest_canonical(
            nib.load(
                f'../data/brainweb_petmr/subject{subject_number:02}/sim_{sim_number}/true_pet.nii.gz'
            ))
        image_3d = cp.array(nii.get_fdata(), dtype=cp.float32)

        # downsample image by a factor of 2, to get 2mm voxels
        image_3d = (image_3d[::2, :, :] + image_3d[1::2, :, :]) / 2
        image_3d = (image_3d[:, ::2, :] + image_3d[:, 1::2, :]) / 2
        image_3d = (image_3d[:, :, ::2] + image_3d[:, :, 1::2]) / 2

        num_axial = 1

        image_shape = (image_3d.shape[0], image_3d.shape[1], 1)

        image_origin = tuple(
            (-0.5 * image_shape[i] + 0.5) * voxel_size[i] for i in range(3))

        # setup the coincidence descriptor
        coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
            num_rings=num_rings,
            sinogram_spatial_axis_order=coincidences.
            SinogramSpatialAxisOrder[sinogram_order],
            radial_trim=171,
            xp=cp)

        # add a subsetter to the projector such that we can do updates with ordered subsets
        subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor,
                                                  num_subsets)

        # update the resolution model which can be different for the reconstruction
        res_model_recon = resolution_models.GaussianImageBasedResolutionModel(
            image_shape, tuple(fwhm_mm_recon / (2.35 * x) for x in voxel_size),
            cp, ndi)

        # loop over slices that are not empty
        z_profile = image_3d.sum((0, 1))
        for sl in cp.where(z_profile > 0.25 * z_profile.max())[0]:
            start_sl = int(sl)
            print(sim_number, subject_number, start_sl)
            image = image_3d[:, :, start_sl:(start_sl + 1)]
            attenuation_image = (
                0.01 * ndi.binary_fill_holes(image.squeeze() > 0)).astype(
                    cp.float32).reshape(image_shape)

            #---------------------------------------------------------------------
            #--- setup of the PET forward model (the projector) ------------------
            #---------------------------------------------------------------------

            #---------------------------------------------------------------------
            # setup a non-time-of-flight and time-of-flight projector
            # to calculate the attenuation factors based on the attenuation image
            projector = petprojectors.PETJosephProjector(
                coincidence_descriptor, image_shape, image_origin, voxel_size)

            # simulate the attenuation factors (exp(-fwd(attenuation_image)))
            attenuation_factors = cp.exp(-projector.forward(attenuation_image))

            # use GE DMI TOF parameters, but only 19 since we are only reconstructing
            # the center of the FOV (brain scan)
            tof_parameters = tof.ge_discovery_mi_tof_parameters
            tof_parameters.num_tofbins = 19
            projector.tof_parameters = tof_parameters

            #--------------------------------------------------------------------------
            # use an image-based resolution model in the projector to model the effect
            # of limited resolution
            res_model = resolution_models.GaussianImageBasedResolutionModel(
                image_shape,
                tuple(fwhm_mm_data / (2.35 * x) for x in voxel_size), cp, ndi)

            projector.image_based_resolution_model = res_model

            #--------------------------------------------------------------------------
            # set the multiplicative corrections in the scanner to model photon attenuation
            # and scanner sensitivity

            projector.multiplicative_corrections = attenuation_factors

            #------------------------------------------------------------------------------
            #------------------------------------------------------------------------------
            #--- simulate acquired data based on forward model and known contaminations ---
            #------------------------------------------------------------------------------
            #------------------------------------------------------------------------------

            image_fwd = projector.forward(image)

            # scale the image such that we get a certain true count per emission voxel value
            emission_volume = cp.where(
                image > 0)[0].shape[0] * np.prod(voxel_size)
            current_trues_per_volume = float(image_fwd.sum() / emission_volume)

            scaling_factor = (trues_per_volume / current_trues_per_volume)
            image *= scaling_factor
            image_fwd *= scaling_factor

            # simulate a constant background contamination
            contamination = cp.full(projector.output_shape,
                                    image_fwd.mean(),
                                    dtype=cp.float32)

            # generate noisy data
            data = cp.random.poisson(image_fwd + contamination).astype(
                cp.uint16)

            #---------------------------------------------------------------------
            #---------------------------------------------------------------------
            #--- run OSEM reconstruction -----------------------------------------
            #---------------------------------------------------------------------
            #---------------------------------------------------------------------

            projector.subsetter = subsetter
            projector.image_based_resolution_model = res_model_recon

            reconstructor = algorithms.OSEM(data,
                                            contamination,
                                            projector,
                                            verbose=False)
            reconstructor.run(num_iterations, evaluate_cost=False)

            x = reconstructor.x

            #---------------------------------------------------------------------
            #---------------------------------------------------------------------
            #--- save the data we need later -------------------------------------
            #---------------------------------------------------------------------
            #---------------------------------------------------------------------

            odir = Path(
                '..'
            ) / 'data' / f'OSEM_2D_{trues_per_volume:.2E}' / f'{subject_number:03}_{sim_number:03}_{start_sl:03}'
            odir.mkdir(parents=True, exist_ok=True)

            # save the image we need
            cp.savez_compressed(odir / 'image.npz', image)
            cp.savez_compressed(odir / f'osem_{seed:03}.npz', x)
            cp.savez_compressed(odir / f'adjoint_ones.npz',
                                reconstructor.adjoint_ones)

            # all sinograms we need
            # to facilitate the handling of subset data, we reshape the data
            # into subsets
            cp.savez_compressed(
                odir / f'data_{seed:03}.npz',
                subsets.split_subset_data(data, projector.subsetter))
            cp.savez_compressed(
                odir / 'image_fwd.npz',
                subsets.split_subset_data(image_fwd, projector.subsetter))
            cp.savez_compressed(
                odir / 'multiplicative_corrections.npz',
                subsets.split_subset_data(projector.multiplicative_corrections,
                                          projector.subsetter))
            cp.savez_compressed(
                odir / 'contamination.npz',
                subsets.split_subset_data(contamination, projector.subsetter))

            # pickle (dill) the projector, but without multiplicative corrections
            projector.multiplicative_corrections = None

            with open(odir / 'projector.pkl', 'wb') as f:
                dill.dump(projector, f)

            with open(odir / 'parameters.json', 'w') as f:
                json.dump(
                    {
                        i: eval(i)
                        for i in [
                            "num_iterations", "num_subsets", "subject_number",
                            "sim_number", "start_sl", "voxel_size",
                            "trues_per_volume", "fwhm_mm_data", "fwhm_mm_recon"
                        ]
                    }, f)