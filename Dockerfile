FROM jupyter/scipy-notebook
MAINTAINER Bas Nijholt <basnijholt@gmail.com>

USER $NB_USER

# Update all because of ipyparallel/ipykernel incompatibilities.
RUN conda update -y --all

# Install packages
RUN conda install -y -c basnijholt kwant==1.2.2
RUN conda install -y -c basnijholt discretizer==0.2
RUN conda install -y -c basnijholt wraparound=1.0
RUN conda install -y holoviews==1.6.2

# Enable ipyparallel
RUN conda install -y ipyparallel
RUN ipython profile create --parallel
RUN ipcluster nbextension enable --user
