import json
from pathlib import Path
import h5py
import pandas as pd
import seaborn as sns

import time

import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.wrapper as wrapper
import pyparallelproj.tof as tof

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

#---------------------------------------------------------------------

xp = np

num_runs = 10
nevents = 10000000
num_subsets = 1
threadsperblock = 64

# image properties
num_trans = 215
num_ax = 71
voxel_size = np.array([2.78, 2.78, 2.78], dtype=np.float32)

# scanner properties
num_rings = 36

# tof parameters
speed_of_light = 300.  # [mm/ns]
time_res_FWHM = 0.385  # [ns]

tof_parameters = tof.TOFParameters(
    num_tofbins=29,
    tofbin_width=13 * 0.01302 * speed_of_light / 2,
    sigma_tof=(speed_of_light / 2) * (time_res_FWHM / 2.355),
    num_sigmas=3)
#---------------------------------------------------------------------
symmetry_axes = (0, 1, 2)

df_fwd = pd.DataFrame()
df_back = pd.DataFrame()

#---------------------------------------------------------------------
# load listmode data
with open('.nema_data.json', 'r') as jfile:
    data_path = Path(json.load(jfile)['path'])

with h5py.File(data_path / 'LIST0000.BLF', 'r') as data:
    events = data['MiceList/TofCoinc'][:]

if nevents is None:
    nevents = events.shape[0]

# shuffle events since events come semi sorted
print('shuffling LM data')
ie = np.arange(nevents)
np.random.shuffle(ie)
events = events[ie, :]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed
events[:, -1] = -(events[:, -1] // 13)

y = xp.ones(events.shape[0], dtype=xp.float32)

for ia, symmetry_axis in enumerate(symmetry_axes):
    image_shape = 3 * [num_trans]
    image_shape[symmetry_axis] = num_ax
    image_shape = tuple(image_shape)
    image_origin = np.array([(-0.5 * image_shape[i] + 0.5) * voxel_size[i]
                             for i in range(3)],
                            dtype=np.float32)

    print(
        f'{symmetry_axis, image_shape} {threadsperblock} tpb  {nevents//1000000}e6 events'
    )

    scanner = scanners.GEDiscoveryMI(num_rings,
                                     symmetry_axis=symmetry_axis,
                                     xp=xp)

    xstart = scanner.get_lor_endpoints(events[:, 0],
                                       events[:, 1]).astype(xp.float32)
    xend = scanner.get_lor_endpoints(events[:, 2],
                                     events[:, 3]).astype(xp.float32)
    tofbin = events[:, 4].astype(xp.int16)
    if xp.__name__ == 'cupy':
        tofbin = xp.asarray(tofbin)

    image = xp.ones(image_shape, dtype=xp.float32)
    image_fwd = xp.zeros(nevents, dtype=xp.float32)
    back_image = xp.zeros(image_shape, dtype=xp.float32)

    for ir in range(num_runs + 1):
        t0 = time.time()
        wrapper.joseph3d_fwd_tof_lm(xstart,
                                    xend,
                                    image,
                                    image_origin,
                                    voxel_size,
                                    image_fwd,
                                    tof_parameters.tofbin_width,
                                    xp.array([tof_parameters.sigma_tof],
                                             dtype=xp.float32),
                                    xp.array([tof_parameters.tofcenter_offset],
                                             dtype=xp.float32),
                                    tof_parameters.num_sigmas,
                                    tofbin,
                                    threadsperblock=threadsperblock)

        if xp.__name__ == 'cupy':
            cp.cuda.Device().synchronize()
        t1 = time.time()
        if ir > 0:
            tmp = pd.DataFrame(
                {
                    'symmetry axis': symmetry_axis,
                    'run': ir,
                    'time (s)': t1 - t0
                },
                index=[0])
            df_fwd = pd.concat((df_fwd, tmp))

        t2 = time.time()
        wrapper.joseph3d_back_tof_lm(xstart,
                                     xend,
                                     back_image,
                                     image_origin,
                                     voxel_size,
                                     y,
                                     tof_parameters.tofbin_width,
                                     xp.array([tof_parameters.sigma_tof],
                                              dtype=xp.float32),
                                     xp.array(
                                         [tof_parameters.tofcenter_offset],
                                         dtype=xp.float32),
                                     tof_parameters.num_sigmas,
                                     tofbin,
                                     threadsperblock=threadsperblock)

        if xp.__name__ == 'cupy':
            cp.cuda.Device().synchronize()
        t3 = time.time()
        if ir > 0:
            tmp = pd.DataFrame(
                {
                    'symmetry axis': symmetry_axis,
                    'run': ir,
                    'time (s)': t3 - t2
                },
                index=[0])
            df_back = pd.concat((df_back, tmp))

#---------------------------------------------------------------------
# plots

df_sum = df_fwd.copy()
df_sum['time (s)'] = df_fwd['time (s)'] + df_back['time (s)']

fig, ax = plt.subplots(1, 3, figsize=(3 * 4, 4), sharex=True, sharey=True)
sns.barplot(data=df_fwd, x='symmetry axis', y='time (s)', ax=ax[0])
sns.barplot(data=df_back, x='symmetry axis', y='time (s)', ax=ax[1])
sns.barplot(data=df_sum, x='symmetry axis', y='time (s)', ax=ax[2])

ax[0].set_title('forward projection')
ax[1].set_title('back projection')
ax[2].set_title('forward + back projection')

for axx in ax.ravel():
    axx.grid(ls=':')

fig.tight_layout()
fig.show()