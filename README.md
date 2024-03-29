# node

A TensorFlow (V2) implementation of [nerual ordinary differential equations](https://arxiv.org/abs/1806.07366).

## TODO

- [X] stable MLP NODE for MNIST dataset.
- [X] Stable CNN NODE for MNIST dataset.
- [X] Generic function that computes the trajectory.
- [X] Visualization utils (How to visualize high dimensional trajectory).
- [X] Generalizing the input type from `tf.Tensor` to `Nest[tf.Tensor]`.
- [X] Study the fix point for the trained stable NODE.
- [X] Adaptive numeric precision in solver.
- [X] CNF.
- [X] Convert to TF code style.
- [X] Add adaptive solvers.
- [X] Re-write ALBERT with NODE.
- [X] Time series forecasting via ALBERT.
- [X] Precise definitions for phase point, phase vector field, e.t.c.
- [ ] Adaptive time interval.
- [ ] Dynamical convergence in Hopfield.
