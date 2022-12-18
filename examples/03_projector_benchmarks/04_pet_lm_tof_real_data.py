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

parser = argparse.ArgumentParser()
parser.add_argument('--num_events', type=int, default=40000000)
parser.add_argument('--num_iterations', type=int, default=6)
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

xp.random.seed(1)

import pyparallelproj.coincidences as coincidences
import pyparallelproj.tof as tof
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.listmode_algorithms as lm_algorithms

data_str = 'nema_tof_listmode'
if presort:
    data_str += '_presorted'

# image properties
num_trans = 215
num_ax = 71
voxel_size = (2.78, 2.78, 2.78)

# scanner properties
num_rings = 36
symmetry_axis = args.symmetry_axis
fwhm_mm_recon = 4.5
tof_parameters = tof.ge_discovery_mi_tof_parameters

# reconstruction parameters
num_iterations = args.num_iterations
num_subsets = args.num_subsets
threadsperblock = args.threadsperblock
num_events = args.num_events

output_dir = args.output_dir
if args.output_file is None:
    output_file = f'{data_str}__mode_{args.mode}__tpb_{threadsperblock}__numevents_{num_events}__axis_{symmetry_axis}.json'

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

# replace zero values in sensitivity image outside scanner by small value
i0 = np.where(adjoint_ones == 0)
i1 = np.where(adjoint_ones > 0)
adjoint_ones[i0] = adjoint_ones[i1].min()

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

# sort events according to in-ring difference
if presort:
    print('pre-sorting events')
    events = events[xp.argsort(events[:, 1] - events[:, 3]), :]

# pass listmode events and multiplicative_correction_list to the projector
projector.events = events
projector.multiplicative_correction_list = multiplicative_correction_list

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- run LM OSEM reconstruction --------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

projector.tof_parameters = tof.ge_discovery_mi_tof_parameters
projector.listmode_subsetter.num_subsets = num_subsets

print('starting reconstruction')
listmode_reconstructor = lm_algorithms.LM_OSEM(contamination_list, projector,
                                               adjoint_ones)
listmode_reconstructor.run(num_iterations)

x_lm = listmode_reconstructor.x

print(f'time per iteration {np.diff(listmode_reconstructor.walltime)}s')

res = {
    'iteration time (s)': np.diff(listmode_reconstructor.walltime).tolist(),
    'iteration time mean (s)': np.diff(listmode_reconstructor.walltime).mean(),
    'iteration time std (s)': np.diff(listmode_reconstructor.walltime).std(),
    'num_iterations': num_iterations,
    'num_subsets': num_subsets,
    'num_events': num_events,
    'presort': presort,
    'mode': args.mode,
    'symmetry axis': symmetry_axis,
    'tpb': threadsperblock,
    'fwhm_mm_recon': fwhm_mm_recon
}

with open(Path(output_dir) / output_file, 'w') as f:
    json.dump(res, f)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---- visualizations -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# get arrays from GPU if running with cupy
if xp.__name__ == 'cupy':
    x_lm = xp.asnumpy(x_lm)

x_lm_sm = gaussian_filter(x_lm, post_sm_fwhm / (2.35 * np.array(voxel_size)))

fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex='col', sharey='col')
ims = dict(cmap=plt.cm.Greys,
           origin='lower',
           vmin=0,
           vmax=0.35 * num_events / 4e7)
ax[0, 0].imshow(np.take(x_lm, 51, axis=symmetry_axis).T, **ims)
ax[0, 1].imshow(
    np.take(x_lm, num_trans // 2, axis=((symmetry_axis + 2) % 3)).T, **ims)
ax[1, 0].imshow(np.take(x_lm_sm, 51, axis=symmetry_axis).T, **ims)
ax[1, 1].imshow(
    np.take(x_lm_sm, num_trans // 2, axis=((symmetry axis + 2) % 3)).T, **ims)
fig.tight_layout()
fig.show()