# Project on Robustness of Majorana bound states in the short junction limit
By Doru Sticlet, Bas Nijholt, and Anton Akhmerov


# Files
This folder contains three IPython notebooks and two Python files:
* `Generate-data.ipynb`
* `Explore-data.ipynb`
* `Paper-figures.ipynb`
* `shortjunction.py`
* `wraparound.py`

Most of the functions used in `Generate-data.ipynb` are defined in `shortjunction.py`.

All notebooks contain instructions of how it can be used.

## Generate-data.ipynb
Generates numerical data used in the paper.

## Explore-data.ipynb
Explore data files uploaded on the 4TU library.

## Paper-figures.ipynb
Plot the numerical figures in the paper


# Data
Download the data used in `Explore-data.ipynb` and `Paper-figures.ipynb` at http://doi.org/10.4121/uuid:274bdd06-14a5-45c3-bc86-87d400082e34


# Installation
Install [miniconda](http://conda.pydata.org/miniconda.html) and then the Python 
environment that contains all dependencies with:

```
conda env create -f environment.yml
```

Run a `jupyter-notebook` to open the `*.ipynb` files.