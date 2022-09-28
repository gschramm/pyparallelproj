import numpy as np
import numpy.typing as npt


def setup_projection_coordinates(num_rad: int,
                                 D: int,
                                 phis: npt.NDArray,
                                 zstart=0,
                                 zend=0) -> tuple[npt.NDArray, npt.NDArray]:

    # setup the start and end coordinates of a single parallel view
    x0 = np.linspace(-D / 2, D / 2, num_rad, dtype=np.float32)
    x1 = np.full(num_rad, D / 2, dtype=np.float32)

    xstart = np.zeros((phis.shape[0], 3 * num_rad), dtype=np.float32)
    xend = np.zeros((phis.shape[0], 3 * num_rad), dtype=np.float32)

    for i, phi in enumerate(phis):
        xstart[i, 0::3] = np.cos(phi) * x0 - np.sin(phi) * x1
        xstart[i, 1::3] = np.sin(phi) * x0 + np.cos(phi) * x1
        xstart[i, 2::3] = zstart

        xend[i, 0::3] = np.cos(phi) * x0 - np.sin(phi) * (-x1)
        xend[i, 1::3] = np.sin(phi) * x0 + np.cos(phi) * (-x1)
        xend[i, 2::3] = zend

    return xstart, xend
