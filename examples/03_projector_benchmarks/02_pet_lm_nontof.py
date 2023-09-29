import time
import argparse
import os
import numpy as np
import json
from pathlib import Path
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--num_events', type=int, default=10000000)
parser.add_argument('--mode', default='GPU', choices=['GPU', 'CPU', 'hybrid'])
parser.add_argument('--threadsperblock', type=int, default=32)
parser.add_argument('--output_file', type=int, default=None)
parser.add_argument('--output_dir', default='results')
parser.add_argument('--presort', action='store_true')
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

import pandas as pd
import parallelproj
import pyparallelproj.scanners as scanners

num_runs = args.num_runs
threadsperblock = args.threadsperblock
num_events = args.num_events

data_str = 'nontof_listmode'
if args.presort:
    data_str += '_presorted'

output_dir = args.output_dir
if args.output_file is None:
    output_file = f'{data_str}__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numevents_{num_events}.csv'

# image properties
num_trans = 215
num_ax = 71
voxel_size = np.array([2.78, 2.78, 2.78], dtype=np.float32)

# scanner properties
num_rings = 36

#---------------------------------------------------------------------
symmetry_axes = (0, 1, 2)

df = pd.DataFrame()

#---------------------------------------------------------------------
# load listmode data
with open('.nema_data.json', 'r') as jfile:
    data_path = Path(json.load(jfile)['path'])

with h5py.File(data_path / 'LIST0000.BLF', 'r') as data:
    events = data['MiceList/TofCoinc'][:]

if num_events is None:
    num_events = events.shape[0]

# shuffle events since events come semi sorted
print('shuffling LM data')
ie = np.arange(num_events)
np.random.shuffle(ie)
events = events[ie, :]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed
events[:, -1] = -(events[:, -1] // 13)

# sort events according to in-ring difference
if args.presort:
    print('pre-sorting events')
    events = events[xp.argsort(events[:, 1] - events[:, 3]), :]

y = xp.ones(events.shape[0], dtype=xp.float32)

for ia, symmetry_axis in enumerate(symmetry_axes):
    image_shape = 3 * [num_trans]
    image_shape[symmetry_axis] = num_ax
    image_shape = tuple(image_shape)
    image_origin = np.array([(-0.5 * image_shape[i] + 0.5) * voxel_size[i]
                             for i in range(3)],
                            dtype=np.float32)

    print(
        f'{symmetry_axis, image_shape} {threadsperblock} tpb  {num_events//1000000}e6 events'
    )

    scanner = scanners.GEDiscoveryMI(num_rings,
                                     symmetry_axis=symmetry_axis,
                                     xp=xp)

    xstart = scanner.get_lor_endpoints(events[:, 0],
                                       events[:, 1]).astype(xp.float32)
    xend = scanner.get_lor_endpoints(events[:, 2],
                                     events[:, 3]).astype(xp.float32)

    image = xp.ones(image_shape, dtype=xp.float32)
    image_fwd = xp.zeros(num_events, dtype=xp.float32)
    back_image = xp.zeros(image_shape, dtype=xp.float32)

    for ir in range(num_runs + 1):
        t0 = time.time()
        img_fwd = parallelproj.joseph3d_fwd(xstart,
                                  xend,
                                  image,
                                  image_origin,
                                  voxel_size,
                                  threadsperblock=threadsperblock)
        t1 = time.time()

        # peform a back projection
        t2 = time.time()
        back_image = parallelproj.joseph3d_back(xstart,
                                   xend,
                                   image_shape,
                                   image_origin,
                                   voxel_size,
                                   y,
                                   threadsperblock=threadsperblock)
        t3 = time.time()
        if ir > 0:
            tmp = pd.DataFrame(
                {
                    'symmetry axis': symmetry_axis,
                    'run': ir,
                    't forward (s)': t1 - t0,
                    't back (s)': t3 - t2
                },
                index=[0])
            df = pd.concat((df, tmp))

#---------------------------------------------------------------------
# save results
df['data'] = data_str
df['mode'] = args.mode
df['num_events'] = num_events
df['threadsperblock'] = threadsperblock

Path(output_dir).mkdir(exist_ok=True, parents=True)
df.to_csv(os.path.join(output_dir, output_file), index=False)