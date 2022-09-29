"""script that show how to TOF forward and backproject numpy arrays using parallelproj in sinogram mode"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyparallelproj.wrapper import joseph3d_fwd_tof_sino, joseph3d_back_tof_sino

from utils import setup_projection_coordinates


def main() -> None:
    voxsize = np.array([4., 4., 4.], dtype=np.float32)
    n0 = 100
    n1 = 100
    n2 = 9

    num_rad = 200
    D = 500
    phi = 1 * np.pi / 8

    # setup an image containing a square
    img = np.zeros((n0, n1, n2), dtype=np.float32)
    img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4),
        (n2 // 4):(3 * n2 // 4)] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (-(np.array(img.shape, dtype=np.float32) / 2) + 0.5) * voxsize

    # setup the start / end coordinates of the LORs we want to project
    # here we use the start / end points of several parallel views of the central direct projection "plane"
    # Note:
    #  1. in general, the start and end point of the LORs can be anywhere in space
    #  2. the C/CUDA projector libs always 1D input arrays,
    #     so all multidim. arrays get "ravelled / flattened" before calling the C/CUDA functions
    phis = np.linspace(0, np.pi, 150, endpoint=False)
    xstart, xend = setup_projection_coordinates(num_rad,
                                                D,
                                                phis,
                                                zstart=0,
                                                zend=0)

    ntofbins = 27
    tofbin_width = 19.
    sigma_tof = np.array([26.], dtype=np.float32)
    tofcenter_offset = np.array([0.], dtype=np.float32)
    nsigmas = 3.

    # setup the array for the forward projection
    img_fwd = np.zeros((xstart.shape[0], num_rad, ntofbins), dtype=np.float32)

    joseph3d_fwd_tof_sino(xstart,
                          xend,
                          img,
                          img_origin,
                          voxsize,
                          img_fwd,
                          tofbin_width=tofbin_width,
                          sigma_tof=sigma_tof,
                          tofcenter_offset=tofcenter_offset,
                          nsigmas=nsigmas,
                          ntofbins=ntofbins)

    # setup array for back projection
    back_img = np.zeros((n0, n1, n2), dtype=np.float32)

    joseph3d_back_tof_sino(xstart,
                           xend,
                           back_img,
                           img_origin,
                           voxsize,
                           img_fwd,
                           tofbin_width=tofbin_width,
                           sigma_tof=sigma_tof,
                           tofcenter_offset=tofcenter_offset,
                           nsigmas=nsigmas,
                           ntofbins=ntofbins)

    # show the results
    ims = dict(cmap=plt.cm.Greys)
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    ax[0].imshow(img[:, :, n2 // 2], **ims)
    ax[1].imshow(img_fwd[:, :, ntofbins // 2 - 4], **ims)
    ax[2].imshow(img_fwd[:, :, ntofbins // 2], **ims)
    ax[3].imshow(img_fwd[:, :, ntofbins // 2 + 4], **ims)
    ax[4].imshow(back_img[:, :, n2 // 2], **ims)
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()