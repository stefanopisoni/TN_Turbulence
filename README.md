# Tensor Network perspective on Turbulent Flows
Study of 3D fully developed turbulent flows (inviscid and incompressible) in a cubic domain using Tensor Network ansatz.\
Corresponding paper on the ArXiv: @@@@@.

The idea is to conduct a comprehensive analysis on the Matrix product States (MPSs) ecnoding of turbulent flows.
In particular, we investigate the compression power of MPSs over a single snapshot as well as their ability in representing fluid correlations in time by developing a 3D time integrator.
For both cases, as quality metrics, we compute the energy spectrum and the divergence of the evctor field (which is supposed to be zero for an incompressible fluid.)

We make use of the MPS-based tensnet library: https://github.com/raghav-tii/tensnet.

## Single snapshot compression
To carry out our analysis on the compressibility of a single turbulent snapshot, we consider the dataset https://torroja.dmt.upm.es/turbdata/Isotropic/.

## 3D solver
The 3D time solver is an extension of the 2D solver presented in the paper https://www.nature.com/articles/s42005-024-01623-8#MOESM2. \
(Note: no mask is required, as the simulations are carried out in a $1024^3$ empty domain.)
