# Compression, simulation and synthesis of turbulent flows with tensor networks
Study of 3D fully developed turbulent flows (inviscid and incompressible) in a cubic domain using Tensor Network ansatz.\
Corresponding paper on the ArXiv: https://arxiv.org/abs/2506.05477.

The idea is to conduct a comprehensive analysis on the Matrix product State (MPS)/Tensor Train (TT) ecnoding of turbulent flows.
In particular, we investigate the compression power of TTs over a single snapshot as well as their ability in representing fluid correlations in time by developing a 3D time integrator.
Moreover, we propose an algorithm to synthetize turbulent fields efficiently  in TTs.

We make use of the TT-based tensnet library: https://github.com/raghav-tii/tensnet.

## Snapshots compression
To carry out our analysis on the compressibility of a single turbulent snapshot, we consider the dataset https://torroja.dmt.upm.es/turbdata/Isotropic/.

## 3D evolution
The 3D time solver is an extension of the 2D solver presented in the paper https://www.nature.com/articles/s42005-024-01623-8#MOESM2. \
(Note: no mask is required, as the simulations are carried out in a $1024^3$ empty domain.)

## Synthesis
This algorithm requires a novel TT interpolation technique soon to be released.
