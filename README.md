# pyparallelproj

python bindings for parallelproj 3D Joseph non-TOF and TOF forward and back projectors.

## Installation

The easiest and recommended way to install the pyparallelproj and all its dependencies is to get them from [our conda channel](https://anaconda.org/gschramm/pyparallelproj) via

```
conda install -c gschramm -c conda-forge pyparallelproj
```

*Remarks*:
- *As usual, we recommend to install this conda package into a separate conda virtual enviornment.* 
- *Currently, a conda package version of the parallelproj libs is only available for ```linux-64```. If you would like to use pyparallelproj on other platforms, you have to build [parallelproj](https://github.com/gschramm/parallelproj) from source and install pyparallelproj from source.*
- *Even if you do not have a CUDA GPU on your system, the compiled CUDA lib (and also the cudatoolkit) gets installed.*  

## Test the installation and run examples

To test whether the python package was installed correctly run the following in python.

```python
import pyparallelproj as ppp
```

To test whether the compiled OpenMP lib is installed correctly run

```python
import pyparallelproj as ppp
print(ppp.config.lib_parallelproj_c)
```

If the CUDA lib was compiled, test the installation via

```python
import pyparallelproj as ppp
print(ppp.config.lib_parallelproj_cuda)
```

In the examples sub directory you can find a few demo scripts that show how to use the projectors. 
```
python 00_PET_grad_prior_sinogram.py
python 01_fwd_back_projection.py
```
When imported, pyparallelproj will test whether a CUDA GPU is available or not and run all projections on the GPU using the CUDA libs if possible.

If you want to explicitely disable all visible GPUs (e.g. to test the OpenMP libraries) or you want to use a specific CUDA device, set the enviroment variable `CUDA_VISIBLE_DEVICES`

If you want to learn how to use pyparallelproj on pytorch tensors, have a look at:
```
python 04_fwd_back_pytorch.py
```
