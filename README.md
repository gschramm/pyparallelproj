# pyparallelproj

python bindings for parallelproj 3D Joseph non-TOF and TOF forward and back projectors.

## Installation

pyparallelproj is not on conda-forge yet - but we expect that
this will happen soon.

Until then, to use pyparallelproj, you have to:

### (1) Clone this

```
git clone git@github.com:gschramm/pyparallelproj.git
```

### (2) Create a conda environment with all dependencies

```
cd pyparallelproj
conda env create -f environment.yml
```

This will create a conda environment called `parallelproj` with all required packages (e.g. the `parallelproj` package containing the compiled OpenMP/CUDA libraries).

### (optional 3) Install cupy

If you would like to use pyparallelproj directly on cupy (or pytorch) GPU arrays, you have to install `cupy` as well.

```
conda install -c conda-forge cupy
```

### (4) Add pyparallelproj to your PYTHONPATH variable

On Unix (bash)

```
export PYTHONPATH=/my/path/to/pyparallelproj
```

## Test the installation

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

## Run examples
