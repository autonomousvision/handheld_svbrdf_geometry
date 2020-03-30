#!/bin/bash

# Copyright (c) 2020 Simon Donné, Max Planck Institute for Intelligent Systems, Tuebingen, Germany
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e # exit on error rather than trying to continue
conda_env_name="BRDF"

# testing whether conda is available
echo -n "Ensuring conda is available...   "
if which conda >/dev/null; then
    echo "OK."
else
    echo "Please make sure a conda version (miniconda or anaconda) is available."
    exit 1
fi
echo ""

# test whether the conda environment already exists, create it if it doesn't
echo "Ensuring conda environment '$conda_env_name' is available... "
echo "If code is not running, always double check whether you have activated it."
# create a conda environment
if conda info --env | grep -w "$conda_env_name" >/dev/null; then
    echo "Already exists."
else    # if the env does not exist.
    echo "Creating minimal environment (this might take a while) ... "
    conda create -n $conda_env_name python=3.7 --yes >/dev/null
    echo "Created successfully."
fi
echo ""

# activate the conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
if conda activate $conda_env_name; then
    echo "Activated the conda environment '$conda_env_name' for this bash script."
else
    echo "Failed to activate the conda environment. Exiting."
    exit 1
fi
echo ""

# test for the various packages and conda install them as required
if conda list | grep -w "pytorch" | grep "1.4" >/dev/null; then
    echo "PyTorch 1.4      > Already installed"
else
    echo "PyTorch 1.4      > Installing now (may take a while)..."
    conda install pytorch=1.4 torchvision cudatoolkit -c pytorch --yes
fi
if conda list | grep -w "tqdm" >/dev/null; then
    echo "TQDM             > Already installed"
else
    echo "TQDM             > Installing now (may take a while)..."
    conda install -c conda-forge tqdm --yes
fi
if conda list | grep -w "scipy" >/dev/null; then
    echo "Scipy            > Already installed"
else
    echo "Scipy            > Installing now (may take a while)..."
    conda install -c anaconda scipy --yes
fi
if conda list | grep -w "scikit-image" >/dev/null; then
    echo "Skimage          > Already installed"
else
    echo "Skimage          > Installing now (may take a while)..."
    conda install -c anaconda scikit-image --yes
fi
if conda list | grep -w "matplotlib" >/dev/null; then
    echo "Matplotlib       > Already installed"
else
    echo "Matplotlib       > Installing now (may take a while)..."
    conda install -c conda-forge matplotlib --yes
fi
if pip list | grep -w "opencv" >/dev/null; then
    echo "OpenCV           > Already installed"
else
    echo "OpenCV           > Installing now (may take a while)..."
    pip install opencv-python
fi
if pip list | grep -w "pymaxflow" >/dev/null; then
    echo "PyMaxFlow        > Already installed"
else
    echo "PyMaxFlow        > Installing now (may take a while)..."
    pip install pymaxflow
fi
if pip list | grep -w "open3d" >/dev/null; then
    echo "Open3D           > Already installed"
else
    echo "Open3D           > Installing now (may take a while)..."
    pip install open3d
fi
echo "Pyrender         > Ensuring requirements"
pip install -r code/thirdparty/pyrender/requirements.txt >/dev/null


# compile the CUDA parts written for this repository
if test "$(find code/utils/TOME/ -maxdepth 1 -name '_implementation.*.so' -print -quit)";  then
    echo "TOME CUDA parts   > Already compiled"
else
    echo "TOME CUDA parts   > Compiling now..."
    cd code/utils/TOME
    python setup.py build_ext --inplace
    cd -
fi
echo ""
