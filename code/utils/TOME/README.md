# Torch for Material Estimation

A pre-compiled C++ and CUDA extension to Torch, containing some code we need in this project.
To compile the library, run

    python setup.py build_ext --inplace

The library can then simply be imported from the code/ directory as

    from utils.TOME as TOME

Credits to (https://github.com/MiguelMonteiro/permutohedral_lattice) for the actual permutohedral lattice code
