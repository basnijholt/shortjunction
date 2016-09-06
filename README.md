# Project on Robustness of Majorana bound states in the short junction limit
By Doru Sticlet, Bas Nijholt, and Anton Akhmerov


This folder contains two IPython notebooks:
* Generate-data.ipynb
* Explore-data.ipynb

Most of the functions used in `Generate-data.ipynb` are defined in `shortjunction.py`.

Both notebooks contain instructions of how it can be used.

## Generate-data.ipynb
Generates numerical data used in the paper.

## Explore-data.ipynb
Explore data files uploaded on the 4TU library.

# Installation
If all package dependencies are met, the notebooks will work in Python 3 without
issues. In case it might not work we have created a Docker image that will start
an environment where everything will work.

First [install Docker](https://docs.docker.com/installation/).

You can either build the image yourself or use a precompiled image (recommended).

To download and run just execute:
```
$ docker run -p 8888:8888 -v /path/to/downloaded/folder/:/home/jovyan/work/ basnijholt/kwant:shortjunction
```

OR build yourself (will take ~20 min to build):
```
$ docker build --tag="basnijholt/kwant:shortjunction" /path/to/downloaded/folder/
```

```
$ docker run -p 8888:8888 -v /path/to/downloaded/folder/:/home/jovyan/work/ basnijholt/kwant:shortjunction
```

Now visit http://localhost:8888/notebooks/shortjunction/

Note: If you are on OS X or Windows, Docker will show a IP address upon opening Docker
use this IP instead of localhost.
