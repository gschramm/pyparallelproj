"""minimal example that shows how to setup a cascaded model with layers combining
   (cupy) projections and 3D convolutions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import dill

from pathlib import Path
from datasets import OSEM2DDataSet
from models import sequential_conv_model, PETVarNet

import pymirc.viewer as pv

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)
parser.add_argument('--input_seed', type=int, default=1)
parser.add_argument('--target_image_name', type=str, default='image.npz')
parser.add_argument('--validation_data_dir', type=str, default=None)
parser.add_argument('--search_pattern', type=str, default=None)
parser.add_argument('--sm_fwhm_mm', type=float, default=6.)
args = parser.parse_args()

checkpoint = torch.load(args.ckpt)

checkpoint_path: Path = Path(args.ckpt)
output_dir: Path = checkpoint_path.parent

with open(output_dir / 'config.json', 'r') as f:
    config = json.load(f)

if args.validation_data_dir is None:
    if 'validation_data_dir' in config:
        args.validation_data_dir = config['validation_data_dir']
    else:
        args.validation_data_dir = '../data/validation/OSEM_2D_5.00E+01'

if args.search_pattern is None:
    if 'validation_search_pattern' in config:
        args.search_pattern = config['validation_search_pattern']
    else:
        args.search_pattern = '047_???_???'

args.num_blocks = config['num_blocks']
args.num_layers = config['num_layers']
args.num_features = config['num_features']

num_blocks: int = args.num_blocks
num_layers: int = args.num_layers
num_features: int = args.num_features
input_seed: int = args.input_seed
target_image_name: str = args.target_image_name
validation_data_dir: str = args.validation_data_dir
search_pattern: str = args.search_pattern
sm_fwhm_mm: float = args.sm_fwhm_mm

dtype = torch.float32
device = torch.device("cuda:0")

#---------------------------------------------------------------------------
#--- setup the data loaders ------------------------------------------------
#---------------------------------------------------------------------------

print(validation_data_dir)

validation_data_set = OSEM2DDataSet(validation_data_dir,
                                    search_pattern=search_pattern)
validation_data_set.seed = input_seed
validation_data_set.target_image_name = target_image_name

dataloader = torch.utils.data.DataLoader(validation_data_set,
                                         batch_size=len(
                                             validation_data_set.dir_list),
                                         drop_last=False,
                                         shuffle=False,
                                         num_workers=5,
                                         pin_memory=True,
                                         pin_memory_device='cuda:0')

#---------------------------------------------------------------------------
#--- load the projector ----------------------------------------------------
#---------------------------------------------------------------------------

# the projector (without multiplicative corrections) should be the same for
# all data sets, so we only restore the one from the first data set
with open(validation_data_set.dir_list[0] / 'projector.pkl', 'rb') as f:
    projector = dill.load(f)

#---------------------------------------------------------------------------
#--- define a simple conv net that maps an image onto an image -------------
#---------------------------------------------------------------------------
conv_net = sequential_conv_model(device=device,
                                 kernel_size=(3, 3, 1),
                                 num_layers=num_layers,
                                 num_features=num_features,
                                 dtype=torch.float32)

model = PETVarNet(projector, conv_net, num_blocks=num_blocks)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    (osem, data, multiplicative_corrections, contamination, adjoint_ones, norm,
     image) = next(iter(dataloader))

    x_fwd = model.forward(osem.to(device), data.to(device),
                          multiplicative_corrections.to(device),
                          contamination.to(device), adjoint_ones.to(device),
                          norm.to(device)).cpu()

# convert to numpy arrays and swap axis for viewing
osem = np.flip(np.swapaxes(np.swapaxes(osem.numpy().squeeze(), 0, 2), 0, 1), 1)
x_fwd = np.flip(np.swapaxes(np.swapaxes(x_fwd.numpy().squeeze(), 0, 2), 0, 1),
                1)
image = np.flip(np.swapaxes(np.swapaxes(image.numpy().squeeze(), 0, 2), 0, 1),
                1)

# unnormalize the images
for i in range(norm.shape[0]):
    osem[..., i] *= float(norm[i])
    x_fwd[..., i] *= float(norm[i])
    image[..., i] *= float(norm[i])

# create a smoothed version of the OSEM image
osem_sm = gaussian_filter(osem, sm_fwhm_mm / (2.35 * 2))

ims = dict(vmin=0, vmax=np.percentile(image, 99.5))
vi = pv.ThreeAxisViewer([osem, osem_sm, x_fwd, image], imshow_kwargs=ims)
