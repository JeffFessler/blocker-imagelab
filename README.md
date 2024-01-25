# ImageLab

[![pipeline status](https://gitlab.eecs.umich.edu/cblocker/imagelab/badges/master/pipeline.svg)](https://gitlab.eecs.umich.edu/cblocker/imagelab/commits/master)
[![coverage report](https://gitlab.eecs.umich.edu/cblocker/imagelab/badges/master/coverage.svg)](https://gitlab.eecs.umich.edu/cblocker/imagelab/commits/master)
[![python: 3.8](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-385/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![lifecycle: unstable](https://img.shields.io/badge/lifecycle-unstable-orange)](/)

This is a library of tools I have developed for my research in image reconstruction and inverse problems, mostly for light field data. It provides a lifi subpackage for working with light field data, an mri subpackage for mri reconstruction, and several general tools.

This package is in early alpha stages. A lot of code refactoring and renaming is still happening.

## Installation

```bash
pip install git+https://gitlab.eecs.umich.edu/cblocker/imagelab.git
```

or for development use the below option 

```bash
git clone <this_repo>
cd imagelab 
pip install -e .
```
The `-e` option is not necessary unless you plan to edit the files, but it makes updating with a `git pull` easier.

## Future Work

+ General
    - ability to run some benchmark algorithms, which may not be included in the git repo, including matlab implementations.
    - watermarking frames with numbers/indexes _in progress_
    - continuous space functions with defined continuous transforms _in progress_
    - Improved show_im function that can take iterables or arrays of images
    - interpolation wrappers around arrays 
    - brightness/contrast
    - hue/saturation
    - white balance
    - bilateral filtering
    - Support for PyTorch Tensors, through a dispatch system built on top of numpy methods
+ Light Field
    - Load Lytro images (.lfp) into 5D-arrays L[v,u,y,x,c]
    - Light field viewing widget, with subaperture selection
    - A light field data loader, which will download datasets and convert to a standard axis convention
    - light transport and ray-transfer matrix simulations
    - zoom, translate camera, etc functions
    - show light-field focal manifold
    - create backdrop for simulated light fields
+ Optimization
    - simplex methods, such as nelder-mead
    - parameter search and estimation tools
+ Misc
    - more MRI tools, including multicoil
    - optical diagram renderer

## Contributors

Cameron Blocker <cameronjblocker@gmail.com>

I would also like to acknowledge Jeff Fessler's irt toolbox and Hung Nien's thesis matlab code which inspired some of this work