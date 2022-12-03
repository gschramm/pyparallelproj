import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pyparallelproj.scanners as scanners
import pyparallelproj.coincidences as coincidences
import pyparallelproj.wrapper as ppw
import pyparallelproj.subsets as subsets

try:
    import cupy as cp
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as cp

#---------------------------------------------------------------------

xp = np

num_runs = 5
num_subsets = 34
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

radial_trim = 65

#---------------------------------------------------------------------
sinogram_orders = ('PVR', 'PRV', 'VPR', 'VRP', 'RPV', 'RVP')
symmetry_axes = (0, 1, 2)

df_fwd = pd.DataFrame()
df_back = pd.DataFrame()

for io, sinogram_order in enumerate(sinogram_orders):
    for ia, symmetry_axis in enumerate(symmetry_axes):
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
        cd = coincidences.RegularPolygonPETCoincidenceDescriptor(
            scanner,
            radial_trim=radial_trim,
            max_ring_difference=scanner.num_rings - 1,
            sinogram_spatial_axis_order=coincidences.
            SinogramSpatialAxisOrder[sinogram_order])

        subsetter = subsets.SingoramViewSubsetter(cd, num_subsets)

        #----------------------------------------------------------------------------
        lors = subsetter.get_subset_indices(0)

        start_mod, start_ind, end_mod, end_ind = cd.get_lor_indices(lors)
        xstart = scanner.get_lor_endpoints(start_mod,
                                           start_ind).astype(xp.float32)
        xend = scanner.get_lor_endpoints(end_mod, end_ind).astype(xp.float32)

        # setup a box like test image
        img_shape = [num_trans, num_trans, num_trans]
        img_shape[symmetry_axis] = max(
            int((ring_positions.max() - ring_positions.min()) /
                voxel_size[symmetry_axis]), 1)
        img_shape = tuple(img_shape)
        n0, n1, n2 = img_shape

        # setup an image containing a square
        img = xp.zeros(img_shape, dtype=np.float32)
        sl = [
            slice(n0 // 4, 3 * n0 // 4, None),
            slice(n1 // 4, 3 * n1 // 4, None),
            slice(n2 // 4, 3 * n2 // 4, None)
        ]

        sl[symmetry_axis] = slice(0, img.shape[symmetry_axis], None)
        sl = tuple(sl)
        img[sl] = 1

        # setup the image origin = the coordinate of the [0,0,0] voxel
        img_origin = (-(np.array(img.shape, dtype=np.float32) / 2) +
                      0.5) * voxel_size
        img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32)

        print(cd.sinogram_spatial_axis_order.name)
        print(symmetry_axis, img_shape)

        for ir in range(num_runs):
            # perform a complete fwd projection
            t0 = time.time()
            ppw.joseph3d_fwd(xstart,
                             xend,
                             img,
                             img_origin,
                             voxel_size,
                             img_fwd,
                             threadsperblock=threadsperblock)
            if xp.__name__ == 'cupy':
                cp.cuda.Device().synchronize()
            t1 = time.time()
            tmp = pd.DataFrame(
                {
                    'sinogram order': cd.sinogram_spatial_axis_order.name,
                    'symmetry axis': symmetry_axis,
                    'run': ir,
                    'time (s)': t1 - t0
                },
                index=[0])
            df_fwd = pd.concat((df_fwd, tmp))

            # perform a complete backprojection
            back_img = xp.zeros(img.shape, dtype=xp.float32)
            ones = xp.ones(img_fwd.shape, dtype=xp.float32)
            t2 = time.time()
            ppw.joseph3d_back(xstart,
                              xend,
                              back_img,
                              img_origin,
                              voxel_size,
                              ones,
                              threadsperblock=threadsperblock)
            if xp.__name__ == 'cupy':
                cp.cuda.Device().synchronize()
            t3 = time.time()
            tmp = pd.DataFrame(
                {
                    'sinogram order': cd.sinogram_spatial_axis_order.name,
                    'symmetry axis': symmetry_axis,
                    'run': ir,
                    'time (s)': t3 - t2
                },
                index=[0])
            df_back = pd.concat((df_back, tmp))

            img_fwd = img_fwd.reshape(subsetter.get_sinogram_subset_shape(0))
            if xp.__name__ == 'cupy':
                img_fwd = cp.asnumpy(img_fwd)
                back_img = cp.asnumpy(back_img)

#----------------------------------------------------------------------------
# plots

df_sum = df_fwd.copy()
df_sum['time (s)'] = df_fwd['time (s)'] + df_back['time (s)']

fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
sns.barplot(data=df_fwd,
            x='sinogram order',
            y='time (s)',
            hue='symmetry axis',
            ax=ax[0])
sns.barplot(data=df_back,
            x='sinogram order',
            y='time (s)',
            hue='symmetry axis',
            ax=ax[1])
sns.barplot(data=df_sum,
            x='sinogram order',
            y='time (s)',
            hue='symmetry axis',
            ax=ax[2])

ax[0].set_title('forward projection')
ax[1].set_title('back projection')
ax[2].set_title('forward + back projection')

for axx in ax.ravel():
    axx.grid(ls=':')

sns.move_legend(ax[0], "upper right", ncol=3)
ax[1].get_legend().remove()
ax[2].get_legend().remove()
fig.tight_layout()
fig.show()