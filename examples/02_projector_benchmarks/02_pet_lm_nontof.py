import json
from pathlib import Path
import h5py

import time

import numpy as np
import matplotlib.pyplot as plt

import pyparallelproj.scanners as scanners
import pyparallelproj.wrapper as wrapper

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

#---------------------------------------------------------------------

xp = np

num_subsets = 1
threadsperblock = 64

# image properties
num_trans = 215
num_ax = 71
voxel_size = np.array([2.78, 2.78, 2.78], dtype=np.float32)

# scanner properties
radius = 0.5 * (744.1 + 2 * 8.51)
num_sides = 34
num_lor_endpoints_per_side = 16
lor_spacing = 4.03125
num_rings = 36
ring_positions = 5.31556 * np.arange(num_rings) + (np.arange(num_rings) //
                                                   9) * 2.8
ring_positions -= 0.5 * ring_positions.max()

#---------------------------------------------------------------------
symmetry_axes = (0, 1, 2)

t_fwd = np.zeros(len(symmetry_axes))
t_back = np.zeros(len(symmetry_axes))

#---------------------------------------------------------------------
# load listmode data
with open('.nema_data.json', 'r') as jfile:
    data_path = Path(json.load(jfile)['path'])

with h5py.File(data_path / 'LIST0000.BLF', 'r') as data:
    events = data['MiceList/TofCoinc'][:]

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
    scanner = scanners.RegularPolygonPETScannerGeometry(
        radius,
        num_sides,
        num_lor_endpoints_per_side,
        lor_spacing,
        num_rings,
        ring_positions,
        symmetry_axis=symmetry_axis,
        xp=xp)

    xstart = scanner.get_lor_endpoints(events[:, 0],
                                       events[:, 1]).astype(xp.float32)
    xend = scanner.get_lor_endpoints(events[:, 2],
                                     events[:, 3]).astype(xp.float32)

    image = xp.ones(image_shape, dtype=xp.float32)
    image_fwd = xp.zeros(nevents, dtype=xp.float32)
    back_image = xp.zeros(image_shape, dtype=xp.float32)

    t0 = time.time()
    wrapper.joseph3d_fwd(xstart,
                         xend,
                         image,
                         image_origin,
                         voxel_size,
                         image_fwd,
                         threadsperblock=threadsperblock)

    if xp.__name__ == 'cupy':
        cp.cuda.Device().synchronize()
    t1 = time.time()
    t_fwd[ia] = t1 - t0
    print(t_fwd[ia])

    t2 = time.time()
    wrapper.joseph3d_back(xstart,
                          xend,
                          back_image,
                          image_origin,
                          voxel_size,
                          y,
                          threadsperblock=threadsperblock)

    if xp.__name__ == 'cupy':
        cp.cuda.Device().synchronize()
    t3 = time.time()
    t_back[ia] = t3 - t2
    print(t_back[ia])

t_total = t_fwd + t_back

#---------------------------------------------------------------------
# plots

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
ax.plot(t_fwd, 'o', label='forward')
ax.plot(t_back, 'o', label='back')
ax.plot(t_total, 'o', label='forward + back')

ax.grid(ls=':')
ax.legend()
ax.set_ylabel('time (s)')
ax.set_xlabel('symmetry axis')

fig.tight_layout()
fig.show()
