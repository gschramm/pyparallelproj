"""script that show how to forward and backproject numpy arrays using parallelproj"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyparallelproj.wrapper import joseph3d_fwd, joseph3d_back

from utils import setup_projection_coordinates

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import numpy as cp
    import numpy.typing as cpt

# variable to decide if we want to project numpy arrays (np) or cupy GPU arrays (cp)
xp = cp


def main() -> None:
    voxsize = xp.array([4., 4., 4.], dtype=xp.float32)
    n0 = 100
    n1 = 100
    n2 = 9

    num_rad = 200
    D = 500
    phi = 1 * xp.pi / 8

    # setup an image containing a square
    img = xp.zeros((n0, n1, n2), dtype=xp.float32)
    img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4),
        (n2 // 4):(3 * n2 // 4)] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (-(xp.array(img.shape, dtype=xp.float32) / 2) + 0.5) * voxsize

    # setup the start / end coordinates of the LORs we want to project
    # here we use the start / end points of several parallel views of the central direct projection "plane"
    # Note:
    #  1. in general, the start and end point of the LORs can be anywhere in space
    #  2. the C/CUDA projector libs always 1D input arrays,
    #     so all multidim. arrays get "ravelled / flattened" before calling the C/CUDA functions
    phis = xp.linspace(0, xp.pi, 150, endpoint=False)
    xstart, xend = setup_projection_coordinates(num_rad,
                                                D,
                                                phis,
                                                zstart=0,
                                                zend=0)

    # setup the array for the forward projection
    img_fwd = xp.zeros((xstart.shape[0], num_rad), dtype=xp.float32)

    # execute the forward projection
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)

    # setup array for back projection
    back_img = xp.zeros((n0, n1, n2), dtype=xp.float32)

    # execute the back projection
    joseph3d_back(xstart, xend, back_img, img_origin, voxsize, img_fwd)

    print(f'img type      {type(img)}')
    print(f'img_fwd type  {type(img_fwd)}')
    print(f'back_img type {type(back_img)}')

    # show the results
    ims = dict(cmap=plt.cm.Greys)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    if xp.__name__ == 'numpy':
        ax[0].imshow(img[:, :, n2 // 2], **ims)
        ax[1].imshow(img_fwd, **ims)
        ax[2].imshow(back_img[:, :, n2 // 2], **ims)
    else:
        ax[0].imshow(xp.asnumpy(img[:, :, n2 // 2]), **ims)
        ax[1].imshow(xp.asnumpy(img_fwd), **ims)
        ax[2].imshow(xp.asnumpy(back_img[:, :, n2 // 2]), **ims)
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()