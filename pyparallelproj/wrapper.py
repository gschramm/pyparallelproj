import ctypes
from pyparallelproj.config import lib_parallelproj_c, lib_parallelproj_cuda, n_visible_gpus
from pyparallelproj.config import joseph3d_fwd_cuda_kernel, joseph3d_back_cuda_kernel, joseph3d_fwd_tof_sino_cuda_kernel, joseph3d_back_tof_sino_cuda_kernel, joseph3d_fwd_tof_lm_cuda_kernel, joseph3d_back_tof_lm_cuda_kernel
import math

import numpy as np
import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import numpy as cp
    import numpy.typing as cpt


def calc_chunks(nLORs, n_chunks):
    """ calculate indices to split an array of length nLORs into n_chunks chunks

        example: splitting an array of length 10 into 3 chunks returns [0,4,7,10]
    """
    rem = nLORs % n_chunks
    div = (nLORs // n_chunks)

    chunks = [0]

    for i in range(n_chunks):
        if i < rem:
            nLORs_chunck = div + 1
        else:
            nLORs_chunck = div

        chunks.append(chunks[i] + nLORs_chunck)

    return chunks


def joseph3d_fwd(xstart: npt.NDArray | cpt.NDArray,
                 xend: npt.NDArray | cpt.NDArray,
                 img: npt.NDArray | cpt.NDArray,
                 img_origin: npt.NDArray | cpt.NDArray,
                 voxsize: npt.NDArray | cpt.NDArray,
                 img_fwd: npt.NDArray | cpt.NDArray,
                 threadsperblock: int = 64,
                 n_chunks: int = 1) -> None:
    """ 3D non-tof Joseph forward projector

    Note
    ----
    This is a python wrapper for the C/CUDA parallelproj 3D non-tof Joseph forward projector.

    If no CUDA GPU is present, the respective function in the C lib is called.

    If a CUDA GPU is present, and the input arrays are numpy arrays, the respective 
    function in the CUDA lib is called which includes memory transfer from host to GPU.

    If a CUDA GPU is present, and the input arrays are cupy arrays, the respective 
    function CUDA kernels are called directly avoiding memory transfer form host to GPU.


    Parameters
    ----------
    xstart : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the start points of the LORs.
        The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    xend : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the end points of the LORs.
        The start coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    img : npt.NDArray | cpt.NDArray
        array of shape [n0,n1,n2] containing the 3D image to be projected.
        The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : npt.NDArray | cpt.NDArray
        array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
        Units are the ones of voxsize.
    voxsize : npt.NDArray | cpt.NDArray
        array [vs0, vs1, vs2] of the voxel sizes
    img_fwd : npt.NDArray | cpt.NDArray
        array of length nLORs (output) used to store the projections
    threadsperblock : int, optional
        threads per block used to launch CUDA kernels, by default 64
    n_chunks : int, optional
        split projection into n_chunks chunks - useful to reduce GPU memory requirement, by default 1
    """
    img_dim = np.array(img.shape, dtype=np.int32)
    nLORs = np.int64(img_fwd.size)

    if n_visible_gpus > 0:
        # we check whether the image to be projected is already on the GPU
        # (cupy array) or whether it is a numpy host array
        # in case the arrays are already on teh GPU, we can call the kernel directly
        if isinstance(img, np.ndarray):
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                ok = lib_parallelproj_cuda.joseph3d_fwd_cuda(
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_img,
                    img_origin, voxsize,
                    img_fwd.ravel()[ic[i]:ic[i + 1]], ic[i + 1] - ic[i],
                    img_dim, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
        else:
            ok = joseph3d_fwd_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), img_fwd, nLORs,
                 cp.asarray(img_dim)))
    else:
        ok = lib_parallelproj_c.joseph3d_fwd(xstart.ravel(), xend.ravel(),
                                             img.ravel(), img_origin, voxsize,
                                             img_fwd.ravel(), nLORs, img_dim)

    return ok


def joseph3d_back(xstart: npt.NDArray | cpt.NDArray,
                  xend: npt.NDArray | cpt.NDArray,
                  back_img: npt.NDArray | cpt.NDArray,
                  img_origin: npt.NDArray | cpt.NDArray,
                  voxsize: npt.NDArray | cpt.NDArray,
                  sino: npt.NDArray | cpt.NDArray,
                  threadsperblock: int = 64,
                  n_chunks: int = 1):
    """ 3D non-tof Joseph back projector

    Note
    ----
    This is a python wrapper for the C/CUDA parallelproj 3D non-tof Joseph back projector.

    If no CUDA GPU is present, the respective function in the C lib is called.

    If a CUDA GPU is present, and the input arrays are numpy arrays, the respective 
    function in the CUDA lib is called which includes memory transfer from host to GPU.

    If a CUDA GPU is present, and the input arrays are cupy arrays, the respective 
    function CUDA kernels are called directly avoiding memory transfer form host to GPU.


    Parameters
    ----------
    xstart : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the start points of the LORs.
        The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    xend : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the end points of the LORs.
        The start coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    back_img : npt.NDArray | cpt.NDArray
        array of shape [n0,n1,n2] containing the 3D image used to store back projection.
        The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : npt.NDArray | cpt.NDArray
        array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
        Units are the ones of voxsize.
    voxsize : npt.NDArray | cpt.NDArray
        array [vs0, vs1, vs2] of the voxel sizes
    sino : npt.NDArray | cpt.NDArray
        array of length nLORs (output) containing the values to be backprojected
    threadsperblock : int, optional
        threads per block used to launch CUDA kernels, by default 64
    n_chunks : int, optional
        split back projection into n_chunks chunks - useful to reduce GPU memory requirement, by default 1
    """
    img_dim = np.array(back_img.shape, dtype=np.int32)
    nLORs = np.int64(sino.size)

    if n_visible_gpus > 0:
        if isinstance(sino, np.ndarray):
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                ok = lib_parallelproj_cuda.joseph3d_back_cuda(
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_back_img,
                    img_origin, voxsize,
                    sino.ravel()[ic[i]:ic[i + 1]], ic[i + 1] - ic[i], img_dim,
                    threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, nvox)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, nvox, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, nvox)
        else:
            ok = joseph3d_back_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), back_img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), sino.ravel(),
                 nLORs, cp.asarray(img_dim)))
    else:
        ok = lib_parallelproj_c.joseph3d_back(xstart.ravel(), xend.ravel(),
                                              back_img.ravel(), img_origin,
                                              voxsize, sino.ravel(), nLORs,
                                              img_dim)

    return ok


def joseph3d_fwd_tof_sino(xstart: npt.NDArray | cpt.NDArray,
                          xend: npt.NDArray | cpt.NDArray,
                          img: npt.NDArray | cpt.NDArray,
                          img_origin: npt.NDArray | cpt.NDArray,
                          voxsize: npt.NDArray | cpt.NDArray,
                          img_fwd: npt.NDArray | cpt.NDArray,
                          tofbin_width: float,
                          sigma_tof: npt.NDArray | cpt.NDArray,
                          tofcenter_offset: npt.NDArray | cpt.NDArray,
                          nsigmas: float,
                          ntofbins: int,
                          threadsperblock: int = 64,
                          n_chunks: int = 1):
    """ 3D tof sinogram Joseph forward projector

    Note
    ----
    This is a python wrapper for the C/CUDA parallelproj 3D tof sinogram Joseph forward projector.

    If no CUDA GPU is present, the respective function in the C lib is called.

    If a CUDA GPU is present, and the input arrays are numpy arrays, the respective 
    function in the CUDA lib is called which includes memory transfer from host to GPU.

    If a CUDA GPU is present, and the input arrays are cupy arrays, the respective 
    function CUDA kernels are called directly avoiding memory transfer form host to GPU.


    Parameters
    ----------
    xstart : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the start points of the LORs.
        The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    xend : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the end points of the LORs.
        The start coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    img : npt.NDArray | cpt.NDArray
        array of shape [n0,n1,n2] containing the 3D image to be projected.
        The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : npt.NDArray | cpt.NDArray
        array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
        Units are the ones of voxsize.
    voxsize : npt.NDArray | cpt.NDArray
        array [vs0, vs1, vs2] of the voxel sizes
    img_fwd : npt.NDArray | cpt.NDArray
        array of length nLORs*ntofbins (output) used to store the projections
    tofbin_width: float
        width of the TOF bins in spatial unit (same units as xstart)
    sigma_tof: npt.NDArray | cpt.NDArray
        sigma of Gaussian TOF kernel in spatial unit (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: npt.NDArray | cpt.NDArray
        center offset of the central TOF bin in spatial unit (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    ntofbins: int
        total number of TOF bins
    threadsperblock : int, optional
        threads per block used to launch CUDA kernels, by default 64
    n_chunks : int, optional
        split projection into n_chunks chunks - useful to reduce GPU memory requirement, by default 1
    """

    img_dim = np.array(img.shape, dtype=np.int32)
    nLORs = np.int64(xstart.size // 3)

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(img, np.ndarray):
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                ok = lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_img,
                    img_origin, voxsize,
                    img_fwd.ravel()[(ntofbins * ic[i]):(ntofbins * ic[i + 1])],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof.ravel()[isig0:isig1],
                    tofcenter_offset.ravel()[ioff0:ioff1], nsigmas, ntofbins,
                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset,
                    threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
        else:
            ok = joseph3d_fwd_tof_sino_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), img_fwd.ravel(),
                 nLORs, cp.asarray(img_dim), np.int16(ntofbins),
                 np.float32(tofbin_width), cp.asarray(sigma_tof.ravel()),
                 cp.asarray(tofcenter_offset.ravel()), np.float32(nsigmas),
                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))

    else:
        ok = lib_parallelproj_c.joseph3d_fwd_tof_sino(
            xstart.ravel(), xend.ravel(), img.ravel(), img_origin, voxsize,
            img_fwd.ravel(), nLORs, img_dim, tofbin_width, sigma_tof.ravel(),
            tofcenter_offset.ravel(), nsigmas, ntofbins,
            lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return ok


def joseph3d_back_tof_sino(xstart: npt.NDArray | cpt.NDArray,
                           xend: npt.NDArray | cpt.NDArray,
                           back_img: npt.NDArray | cpt.NDArray,
                           img_origin: npt.NDArray | cpt.NDArray,
                           voxsize: npt.NDArray | cpt.NDArray,
                           sino: npt.NDArray | cpt.NDArray,
                           tofbin_width: float,
                           sigma_tof: npt.NDArray | cpt.NDArray,
                           tofcenter_offset: npt.NDArray | cpt.NDArray,
                           nsigmas: float,
                           ntofbins: int,
                           threadsperblock: int = 64,
                           n_chunks: int = 1):
    """ 3D tof sinogram Joseph back projector

    Note
    ----
    This is a python wrapper for the C/CUDA parallelproj 3D tof sinogram Joseph back projector.

    If no CUDA GPU is present, the respective function in the C lib is called.

    If a CUDA GPU is present, and the input arrays are numpy arrays, the respective 
    function in the CUDA lib is called which includes memory transfer from host to GPU.

    If a CUDA GPU is present, and the input arrays are cupy arrays, the respective 
    function CUDA kernels are called directly avoiding memory transfer form host to GPU.


    Parameters
    ----------
    xstart : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the start points of the LORs.
        The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    xend : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the end points of the LORs.
        The start coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    back_img : npt.NDArray | cpt.NDArray
        array of shape [n0,n1,n2] containing the 3D image used to store back projection.
        The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : npt.NDArray | cpt.NDArray
        array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
        Units are the ones of voxsize.
    voxsize : npt.NDArray | cpt.NDArray
        array [vs0, vs1, vs2] of the voxel sizes
    sino : npt.NDArray | cpt.NDArray
        array of size nLORs*ntofbins (output) containing the values to be backprojected
    tofbin_width: float
        width of the TOF bins in spatial unit (same units as xstart)
    sigma_tof: npt.NDArray | cpt.NDArray
        sigma of Gaussian TOF kernel in spatial unit (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: npt.NDArray | cpt.NDArray
        center offset of the central TOF bin in spatial unit (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: int
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    ntofbins: int
        total number of TOF bins
    threadsperblock : int, optional
        threads per block used to launch CUDA kernels, by default 64
    n_chunks : int, optional
        split projection into n_chunks chunks - useful to reduce GPU memory requirement, by default 1
    """
    img_dim = np.array(back_img.shape, dtype=np.int32)
    nLORs = np.int64(xstart.size // 3)

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(sino, np.ndarray):
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                ok = lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_back_img,
                    img_origin, voxsize,
                    sino.ravel()[(ntofbins * ic[i]):(ntofbins * ic[i + 1])],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof.ravel()[isig0:isig1],
                    tofcenter_offset.ravel()[ioff0:ioff1], nsigmas, ntofbins,
                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset,
                    threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, nvox)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, nvox, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, nvox)
        else:
            ok = joseph3d_back_tof_sino_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), back_img,
                 cp.asarray(img_origin), cp.asarray(voxsize), sino.ravel(),
                 np.int64(nLORs), cp.asarray(img_dim), np.int16(ntofbins),
                 np.float32(tofbin_width), cp.asarray(sigma_tof).ravel(),
                 cp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
    else:
        ok = lib_parallelproj_c.joseph3d_back_tof_sino(
            xstart.ravel(), xend.ravel(), back_img.ravel(), img_origin,
            voxsize, sino.ravel(), nLORs, img_dim, tofbin_width,
            sigma_tof.ravel(), tofcenter_offset.ravel(), nsigmas, ntofbins,
            lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return ok


def joseph3d_fwd_tof_lm(xstart: npt.NDArray | cpt.NDArray,
                        xend: npt.NDArray | cpt.NDArray,
                        img: npt.NDArray | cpt.NDArray,
                        img_origin: npt.NDArray | cpt.NDArray,
                        voxsize: npt.NDArray | cpt.NDArray,
                        img_fwd: npt.NDArray | cpt.NDArray,
                        tofbin_width: float,
                        sigma_tof: npt.NDArray | cpt.NDArray,
                        tofcenter_offset: npt.NDArray | cpt.NDArray,
                        nsigmas: float,
                        tofbin: npt.NDArray | cpt.NDArray,
                        threadsperblock: int = 64,
                        n_chunks: int = 1):
    """ 3D tof listmode Joseph forward projector

    Note
    ----
    This is a python wrapper for the C/CUDA parallelproj 3D tof listmode Joseph forward projector.

    If no CUDA GPU is present, the respective function in the C lib is called.

    If a CUDA GPU is present, and the input arrays are numpy arrays, the respective 
    function in the CUDA lib is called which includes memory transfer from host to GPU.

    If a CUDA GPU is present, and the input arrays are cupy arrays, the respective 
    function CUDA kernels are called directly avoiding memory transfer form host to GPU.


    Parameters
    ----------
    xstart : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the start points of the LORs.
        The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    xend : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the end points of the LORs.
        The start coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    img : npt.NDArray | cpt.NDArray
        array of shape [n0,n1,n2] containing the 3D image to be projected.
        The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : npt.NDArray | cpt.NDArray
        array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
        Units are the ones of voxsize.
    voxsize : npt.NDArray | cpt.NDArray
        array [vs0, vs1, vs2] of the voxel sizes
    img_fwd : npt.NDArray | cpt.NDArray
        array of length nLORs (output) used to store the projections
    tofbin_width: float
        width of the TOF bins in spatial unit (same units as xstart)
    sigma_tof: npt.NDArray | cpt.NDArray
        sigma of Gaussian TOF kernel in spatial unit (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: npt.NDArray | cpt.NDArray
        center offset of the central TOF bin in spatial unit (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    tofbin: npt.NDArray | cpt.NDArray
        arrays containing the tof bin of the events
    threadsperblock : int, optional
        threads per block used to launch CUDA kernels, by default 64
    n_chunks : int, optional
        split projection into n_chunks chunks - useful to reduce GPU memory requirement, by default 1
    """

    img_dim = np.array(img.shape, dtype=np.int32)
    nLORs = np.int64(xstart.size // 3)

    lor_dependent_sigma_tof = int(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = int(tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(img, np.ndarray):
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                ok = lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_img,
                    img_origin, voxsize, img_fwd[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof[isig0:isig1], tofcenter_offset[ioff0:ioff1],
                    nsigmas, tofbin[ic[i]:ic[i + 1]], lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
        else:
            ok = joseph3d_fwd_tof_lm_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), img_fwd,
                 np.int64(nLORs), cp.asarray(img_dim),
                 np.float32(tofbin_width), cp.asarray(sigma_tof),
                 cp.asarray(tofcenter_offset), np.float32(nsigmas), tofbin,
                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
    else:
        ok = lib_parallelproj_c.joseph3d_fwd_tof_lm(
            xstart.ravel(), xend.ravel(), img.ravel(), img_origin, voxsize,
            img_fwd, nLORs, img_dim, tofbin_width, sigma_tof, tofcenter_offset,
            nsigmas, tofbin, lor_dependent_sigma_tof,
            lor_dependent_tofcenter_offset)

    return ok


# ------------------


def joseph3d_back_tof_lm(xstart: npt.NDArray | cpt.NDArray,
                         xend: npt.NDArray | cpt.NDArray,
                         back_img: npt.NDArray | cpt.NDArray,
                         img_origin: npt.NDArray | cpt.NDArray,
                         voxsize: npt.NDArray | cpt.NDArray,
                         lst: npt.NDArray | cpt.NDArray,
                         tofbin_width: float,
                         sigma_tof: npt.NDArray | cpt.NDArray,
                         tofcenter_offset: npt.NDArray | cpt.NDArray,
                         nsigmas: int,
                         tofbin: npt.NDArray | cpt.NDArray,
                         threadsperblock: int = 64,
                         n_chunks: int = 1):
    """ 3D tof listmode Joseph back projector

    Note
    ----
    This is a python wrapper for the C/CUDA parallelproj 3D tof listmode Joseph back projector.

    If no CUDA GPU is present, the respective function in the C lib is called.

    If a CUDA GPU is present, and the input arrays are numpy arrays, the respective 
    function in the CUDA lib is called which includes memory transfer from host to GPU.

    If a CUDA GPU is present, and the input arrays are cupy arrays, the respective 
    function CUDA kernels are called directly avoiding memory transfer form host to GPU.


    Parameters
    ----------
    xstart : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the start points of the LORs.
        The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    xend : npt.NDArray | cpt.NDArray
        array of shape [3*nLORs] with the coordinates of the end points of the LORs.
        The start coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2. 
        Units are the ones of voxsize.
    back_img : npt.NDArray | cpt.NDArray
        array of shape [n0,n1,n2] containing the 3D image used to store back projection.
        The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : npt.NDArray | cpt.NDArray
        array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
        Units are the ones of voxsize.
    voxsize : npt.NDArray | cpt.NDArray
        array [vs0, vs1, vs2] of the voxel sizes
    lst : npt.NDArray | cpt.NDArray
        array of size nLORs (output) containing the values to be backprojected
    tofbin_width: float
        width of the TOF bins in spatial unit (same units as xstart)
    sigma_tof: npt.NDArray | cpt.NDArray
        sigma of Gaussian TOF kernel in spatial unit (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: npt.NDArray | cpt.NDArray
        center offset of the central TOF bin in spatial unit (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: int
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    tofbin: npt.NDArray | cpt.NDArray
        arrays containing the tof bin of the events
    threadsperblock : int, optional
        threads per block used to launch CUDA kernels, by default 64
    n_chunks : int, optional
        split projection into n_chunks chunks - useful to reduce GPU memory requirement, by default 1
    """

    img_dim = np.array(back_img.shape, dtype=np.int32)
    nLORs = np.int64(xstart.size // 3)

    lor_dependent_sigma_tof = int(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = int(tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(lst, np.ndarray):
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                ok = lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_back_img,
                    img_origin, voxsize, lst[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof[isig0:isig1], tofcenter_offset[ioff0:ioff1],
                    nsigmas, tofbin[ic[i]:ic[i + 1]], lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, nvox)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, nvox, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, nvox)
        else:
            lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
            lor_dependent_tofcenter_offset = np.uint8(
                tofcenter_offset.shape[0] == nLORs)

            ok = joseph3d_back_tof_lm_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), back_img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), lst,
                 np.int64(nLORs), cp.asarray(img_dim),
                 np.float32(tofbin_width), cp.asarray(sigma_tof),
                 cp.asarray(tofcenter_offset), np.float32(nsigmas), tofbin,
                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
    else:
        ok = lib_parallelproj_c.joseph3d_back_tof_lm(
            xstart.ravel(), xend.ravel(), back_img.ravel(), img_origin,
            voxsize, lst, nLORs, img_dim, tofbin_width, sigma_tof,
            tofcenter_offset, nsigmas, tofbin, lor_dependent_sigma_tof,
            lor_dependent_tofcenter_offset)

    return ok
