import numpy as np
import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import numpy as cp
    import numpy.typing as cpt


def setup_sinogram_projection_coordinates(
        num_rad: int,
        D: int,
        phis: npt.NDArray | npt.NDArray,
        zstart=0,
        zend=0) -> tuple[npt.NDArray | cpt.NDArray, npt.NDArray | cpt.NDArray]:

    if isinstance(phis, np.ndarray):
        xp = np
    else:
        xp = cp

    # setup the start and end coordinates of a single parallel view
    x0 = xp.linspace(-D / 2, D / 2, num_rad, dtype=xp.float32)
    x1 = xp.full(num_rad, D / 2, dtype=xp.float32)

    xstart = xp.zeros((phis.shape[0], 3 * num_rad), dtype=xp.float32)
    xend = xp.zeros((phis.shape[0], 3 * num_rad), dtype=xp.float32)

    for i, phi in enumerate(phis):
        xstart[i, 0::3] = xp.cos(phi) * x0 - xp.sin(phi) * x1
        xstart[i, 1::3] = xp.sin(phi) * x0 + xp.cos(phi) * x1
        xstart[i, 2::3] = zstart

        xend[i, 0::3] = xp.cos(phi) * x0 - xp.sin(phi) * (-x1)
        xend[i, 1::3] = xp.sin(phi) * x0 + xp.cos(phi) * (-x1)
        xend[i, 2::3] = zend

    return xstart, xend


def setup_listmode_projection_coordinates(
    n: int, D: int, L: int, ntofbins: int, module_name: str
) -> tuple[npt.NDArray | cpt.NDArray, npt.NDArray | cpt.NDArray, npt.NDArray
           | cpt.NDArray]:

    if module_name == 'numpy':
        xp = np
    else:
        xp = cp

    # setup the start and end coordinates of a single parallel view
    phi_start = 2 * xp.pi * xp.random.rand(n)
    z_start = L * xp.random.rand(n) - L / 2
    phi_end = 2 * xp.pi * xp.random.rand(n)
    z_end = L * xp.random.rand(n) - L / 2

    xstart = xp.zeros(3 * n, dtype=xp.float32)
    xend = xp.zeros(3 * n, dtype=xp.float32)

    xstart[0::3] = D * xp.cos(phi_start)
    xstart[1::3] = D * xp.sin(phi_start)
    xstart[2::3] = z_start

    xend[0::3] = D * xp.cos(phi_end)
    xend[1::3] = D * xp.sin(phi_end)
    xend[2::3] = z_end

    tofbin = xp.random.randint(0, ntofbins, n).astype(np.int16)

    return xstart, xend, tofbin
