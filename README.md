# nn2++: Training two-layer neural networks

This small package provides utilities to simulate learning in fully connected
two-layer neural networks. It was used to perform all the experiments of our
recent paper on the dynamics of stochastic gradient descent for two-layer neural
networks in the teacher-student setup [1].

## Install & usage information

To build the simulator, simply type
```
make nn2pp.exe
```
which will create a file ``nn2pp.exe``. 
Run ``./nn2pp.exe -h`` for usage information.

Likewise, type
```
make nn2pp_ode_erf.exe
```
to build the ODE integrator. 
Run ``./nn2pp_ode_erf.exe -h`` for usage information.

## Requirements

* All linear algebra operations are implemented using
  [Armadillo](http://arma.sourceforge.net/), a C++ library for linear algebra &
  scientific computing
* Unit tests are implemented using [Google
  Test](https://github.com/abseil/googletest)

## References

* [1] S. Goldt, M.S. Advani, A.M. Saxe, F. Krzakala, L. Zdeborová, NeurIPS 2019
  (forthcoming), [arXiv:1906.08632](https://arxiv.org/abs/1906.08632)
