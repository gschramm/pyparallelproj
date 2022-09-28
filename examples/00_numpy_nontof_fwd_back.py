"""script that show how to forward and backproject numpy arrays using parallelproj"""

#TODO: - remove shape from wrapper
#      - remove ravel in backprojector

import numpy as np
import numpy.typing as npt
from pyparallelproj.wrapper import joseph3d_fwd, joseph3d_back


def setup_view_coordimates(
        nLORs: int = 300,
        D: int = 600,
        phi: float = 1 * np.pi / 8) -> tuple[npt.NDArray, npt.NDArray]:

    # setup the start and end coordinates of a single parallel view
    x0 = np.linspace(-D / 2, D / 2, nLORs, dtype=np.float32)
    x1 = np.full(nLORs, D / 2, dtype=np.float32)

    xstart = np.zeros(3 * nLORs, dtype=np.float32)
    xstart[0::3] = np.cos(phi) * x0 - np.sin(phi) * x1
    xstart[1::3] = np.sin(phi) * x0 + np.cos(phi) * x1
    xstart[2::3] = np.zeros(nLORs, dtype=np.float32)

    xend = np.zeros(3 * nLORs, dtype=np.float32)
    xend[0::3] = np.cos(phi) * x0 - np.sin(phi) * (-x1)
    xend[1::3] = np.sin(phi) * x0 + np.cos(phi) * (-x1)
    xend[2::3] = np.zeros(nLORs, dtype=np.float32)

    return xstart, xend


if __name__ == "__main__":

    voxsize = np.array([4., 4., 4.], dtype=np.float32)
    n0 = 100
    n1 = 100
    n2 = 10

    nLORs = 150
    D = 600
    phi = 1 * np.pi / 8

    # setup a sqaure image
    img = np.zeros((n0, n1, n2), dtype=np.float32)
    img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4),
        (n2 // 4):(3 * n2 // 4)] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (-(np.array(img.shape, dtype=np.float32) / 2) + 0.5) * voxsize


    # setup the start / end coordinates of the LORs we want to project
    xstart, xend = setup_view_coordimates(nLORs, D, phi)

    # setup the array for the forward projection
    img_fwd = np.zeros(nLORs, dtype=np.float32)

    # execute the forward projection
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs,
                 np.array(img.shape, dtype=np.int32))

    # setup array for back projection
    bimg = np.zeros((n0, n1, n2), dtype=np.float32)
    joseph3d_back(xstart, xend, bimg.ravel(), img_origin, voxsize, img_fwd,
                  nLORs, np.array(img.shape, dtype=np.int32))
