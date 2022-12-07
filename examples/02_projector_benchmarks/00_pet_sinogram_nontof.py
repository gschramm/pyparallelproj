import time
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--num_subsets', type=int, default=34)
parser.add_argument('--mode', default='GPU', choices=['GPU', 'CPU', 'hybrid'])
parser.add_argument('--threadsperblock', type=int, default=64)
parser.add_argument('--output_file', type=int, default=None)
parser.add_argument('--output_dir', default='results')
args = parser.parse_args()

if args.mode == 'GPU':
    import cupy as cp
    xp = cp
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.mode == 'hybrid':
    xp = np
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.mode == 'CPU':
    xp = np
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    raise ValueError

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyparallelproj.coincidences as coincidences
import pyparallelproj.wrapper as wrapper
import pyparallelproj.subsets as subsets

num_runs = args.num_runs
threadsperblock = args.threadsperblock
num_subsets = args.num_subsets

output_dir = args.output_dir
if args.output_file is None:
    output_file = f'tofsinogram__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numsubsets_{num_subsets}.csv'

# image properties
num_trans = 215
num_ax = 71
voxel_size = np.array([2.78, 2.78, 2.78], dtype=np.float32)

# scanner properties
num_rings = 36

#---------------------------------------------------------------------
sinogram_orders = ('PVR', 'PRV', 'VPR', 'VRP', 'RPV', 'RVP')
symmetry_axes = (0, 1, 2)

df = pd.DataFrame()

for io, sinogram_order in enumerate(sinogram_orders):
    for ia, symmetry_axis in enumerate(symmetry_axes):
        # define the coincidence descriptor and scanner
        coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
            num_rings=num_rings,
            sinogram_spatial_axis_order=coincidences.
            SinogramSpatialAxisOrder[sinogram_order],
            symmetry_axis=symmetry_axis,
            xp=xp)

        subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor,
                                                  num_subsets)

        #----------------------------------------------------------------------------
        lors = subsetter.get_subset_indices(0)

        start_mod, start_ind, end_mod, end_ind = coincidence_descriptor.get_lor_indices(
            lors)
        xstart = coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(xp.float32)
        xend = coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(xp.float32)

        # setup a box like test image
        img_shape = [num_trans, num_trans, num_trans]
        img_shape[symmetry_axis] = num_ax
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

        print(coincidence_descriptor.sinogram_spatial_axis_order.name)
        print(symmetry_axis, img_shape)

        for ir in range(num_runs + 1):
            # perform a complete fwd projection
            t0 = time.time()
            wrapper.joseph3d_fwd(xstart,
                                 xend,
                                 img,
                                 img_origin,
                                 voxel_size,
                                 img_fwd,
                                 threadsperblock=threadsperblock)
            t1 = time.time()
            if ir > 0:
                tmp = pd.DataFrame(
                    {
                        'sinogram order':
                        coincidence_descriptor.sinogram_spatial_axis_order.
                        name,
                        'symmetry axis':
                        symmetry_axis,
                        'direction':
                        'forward',
                        'run':
                        ir,
                        'time (s)':
                        t1 - t0
                    },
                    index=[0])
                df = pd.concat((df, tmp))

            # perform a complete backprojection
            back_img = xp.zeros(img.shape, dtype=xp.float32)
            ones = xp.ones(img_fwd.shape, dtype=xp.float32)
            t2 = time.time()
            wrapper.joseph3d_back(xstart,
                                  xend,
                                  back_img,
                                  img_origin,
                                  voxel_size,
                                  ones,
                                  threadsperblock=threadsperblock)
            t3 = time.time()
            if ir > 0:
                tmp = pd.DataFrame(
                    {
                        'sinogram order':
                        coincidence_descriptor.sinogram_spatial_axis_order.
                        name,
                        'symmetry axis':
                        symmetry_axis,
                        'direction':
                        'back',
                        'run':
                        ir,
                        'time (s)':
                        t3 - t2
                    },
                    index=[0])
                df = pd.concat((df, tmp))

#----------------------------------------------------------------------------
# save results

df.to_csv(os.path.join(output_dir, output_file), index=False)

# plots

df_sum = df.loc[df.direction == 'forward'].copy()
df_sum['time (s)'] = df.loc[df.direction == 'forward']['time (s)'] + df.loc[
    df.direction == 'back']['time (s)']
df_sum['direction'] = 'forward+back'

fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
sns.barplot(data=df.loc[df.direction == 'forward'],
            x='sinogram order',
            y='time (s)',
            hue='symmetry axis',
            ax=ax[0])
sns.barplot(data=df.loc[df.direction == 'back'],
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