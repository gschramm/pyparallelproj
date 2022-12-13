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

To test whether the `pyparallelproj` python package can be imported
and whether the compiled `parallelproj` OpenMP library was installed
run:

```python
import pyparallelproj
import pyparallelproj.config as config
```

If you run this on a system without CUDA, it should print sth like:

```
.../pyparallelproj/pyparallelproj/config.py:29: UserWarning: CUDA not available
  warn('CUDA not available')
using PARALLELPROJ_C_LIB .../miniforge/base/envs/parallelproj/bin/../lib/libparallelproj_c.so
```

On a system with CUDA, it should print

```
using PARALLELPROJ_C_LIB .../miniforge/base/envs/parallelproj/bin/../lib/libparallelproj_c.so
using PARALLELPROJ_CUDA_LIB .../miniforge/base/envs/parallelproj/bin/../lib/libparallelproj_cuda.so
```

## Run examples

The examples contrains a few educational examples that show how to use `pyparallelproj`
on different levels.

- `00_projections_and_reconstruction` contains high level examples on how to do
  nonTOF and TOF projections and how to run simple reconstructions
- `01_defining_scanner_geometries` shows how to define custom PET scanners
- `02_low_level_parallelproj_api` shows how to use the `parallelproj` API on a
  lower level

**Note**: To run certain examples, you need to install extra python packages (e.g. h5py, nibabel, pandas).
All of those are available on conda-forge, such that they can be installed via

```
conda install conda-forge mypackage-of-choice
```
