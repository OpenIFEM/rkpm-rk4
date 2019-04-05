# rkpm-rk4

rkpm-rk4 is a hypoelastic solid dynamics simulation code that uses Reproducing Kernel Particle Method (RKPM) in Total Lagrangian weak form for spatial discretization and 4th order Runge-Kutta method (RK4) for time discretization. rkpm-rk4 is adapted from [mroethli/mfree_iwf](https://github.com/mroethli/mfree_iwf) with the following major changes:

* rkpm-rk4 runs in both 2D and 3D.
* rkpm-rk4 supports Neumann boundary condition.
* rkpm-rk4 is dedicated to RKPM and FEM.
* rkpm-rk4 does not support contact modeling among multiple groups of particles.

To understand the theory of RKPM, please refer to *Meshless Methods for Large Deformation Elastodynamics* which can be retrieved from [arxiv](https://arxiv.org/abs/1807.01117), and references listed in mfree_iwf.

To build and install rkpm-rk4:

1. Download [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page). You don't have to *install* it because it is a header-only library.

2. ``cmake <path-to-source-code> -DEIGEN3_INCLUDE_DIR=<path-to-eigen> [-DCMAKE_BUILD_TYPE=(Release|Debug)]``

3. ``make``

4. ``make install``

To run benchmark cases:

``ctest -R <name-of-benchmark> -V``

The output files are in Legacy VTK formt which can be visualized with [ParaView](https://www.paraview.org/).

rkpm-rk4 is free software and licensed under GPLv3.
