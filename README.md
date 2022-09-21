# pyparallelproj

python bindings for parallelproj 3D Joseph non-TOF and TOF forward and back projectors.

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

In the examples sub directory you can find a few demo script that show how to use the projectors. Good examples to start with are `fwd_back.py`, `tof_pet_sino.py` and `tof_pet_lm.py` which demonstrate a simple forward and back projection, a short sinogram and listmode OS-MLEM reconstruction on simulated data. You can run them via

```
python fwd_back.py
python tof_pet_sino.py
python tof_pet_lm.py
```

When imported, pyparallelproj will test whether a CUDA GPU is available or not and run all projections on the GPU using the CUDA libs if possible.

If you want to explicitely disable all visible GPUs (e.g. to test the OpenMP libraries) or you want to use a specific CUDA device, set the enviroment variable `CUDA_VISIBLE_DEVICES`
