"""script that show how to TOF forward and backproject numpy arrays using parallelproj in sinogram mode"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyparallelproj.wrapper import joseph3d_fwd_tof_lm, joseph3d_back_tof_lm

from utils import setup_listmode_projection_coordinates

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import numpy as cp
    import numpy.typing as cpt

# variable to decide if we want to project numpy arrays (np) or cupy GPU arrays (cp)
xp = np


def main() -> None:

    voxsize = xp.array([4., 4., 4.], dtype=xp.float32)
    n0 = 100
    n1 = 100
    n2 = 9

    D = 500

    # setup an image containing a square
    img = xp.zeros((n0, n1, n2), dtype=xp.float32)
    img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4),
        (n2 // 4):(3 * n2 // 4)] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (-(xp.array(img.shape, dtype=xp.float32) / 2) + 0.5) * voxsize

    # setup the start / end coordinates of the LORs we want to project
    # here we use the start / end points randomly distributed on a cylinder surface

    ntofbins = 27
    tofbin_width = 19.
    sigma_tof = xp.array([26.], dtype=xp.float32)
    tofcenter_offset = xp.array([0.], dtype=xp.float32)
    nsigmas = 3.

    nevents = 1000000

    xstart, xend, tofbin = setup_listmode_projection_coordinates(
        nevents, D, n2 * voxsize[2], ntofbins, xp.__name__)

    # setup the array for the forward projection
    img_fwd = xp.zeros(nevents, dtype=xp.float32)

    joseph3d_fwd_tof_lm(xstart, xend, img, img_origin, voxsize, img_fwd,
                        tofbin_width, sigma_tof, tofcenter_offset, nsigmas,
                        tofbin)

    # setup array for back projection
    back_img = xp.zeros((n0, n1, n2), dtype=xp.float32)
    joseph3d_back_tof_lm(xstart, xend, back_img, img_origin, voxsize, img_fwd,
                         tofbin_width, sigma_tof, tofcenter_offset, nsigmas,
                         tofbin)

    print(f'img type      {type(img)}')
    print(f'img_fwd type  {type(img_fwd)}')
    print(f'back_img type {type(back_img)}')

    # show the results
    ims = dict(cmap=plt.cm.Greys)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    if xp.__name__ == 'numpy':
        ax[0].imshow(img[:, :, n2 // 2], **ims)
        ax[1].imshow(back_img[:, :, n2 // 2], **ims)
    else:
        ax[0].imshow(xp.asnumpy(img[:, :, n2 // 2]), **ims)
        ax[1].imshow(xp.asnumpy(back_img[:, :, n2 // 2]), **ims)
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()