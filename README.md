# node

Neural ODE.

## TODO

- [X] stable MLP NODE for MNIST dataset.
- [ ] Stable CNN NODE for MNIST dataset.
- [X] Generic function that computes the trajectory
- [X] Visualization utils (How to visualize high dimensional trajectory).
- [ ] Study the fix point for the trained stable NODE.
- [ ] Generalizing the input type from `tf.Tensor` to `Sequence[tf.Tensor]`.
- [ ] Adaptive time interval.
- [ ] Add adaptive solvers.
- [ ] Convert to TF code style.
- [ ] Re-write ALBERT with NODE.

## Notes

1. `tf.custom_gradient` has an unfixed [bug](https://github.com/tensorflow/tensorflow/issues/31945). Because of this, currently, `node` does NOT support multi-inputs.
