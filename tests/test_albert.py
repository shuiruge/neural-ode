import tensorflow as tf
from argparse import ArgumentParser
from node.applications.nlp.albert import SelfAttention, get_albert_dynamics
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver


PARSER = ArgumentParser()
PARSER.add_argument('--d_model', type=int, default=8)
PARSER.add_argument('--num_heads', type=int, default=4)
PARSER.add_argument('--dff', type=int, default=128)
PARSER.add_argument('--solver', type=str, default='rk4',
                    help='in ("rk4", "rkf56")')
ARGS = PARSER.parse_args()


def get_self_attention(d_model, num_heads):

  _self_attention = SelfAttention(d_model, num_heads)

  def self_attention(x, mask):
    return _self_attention([x, mask])

  return self_attention


def get_feed_forward(d_model, dff):

  hidden_layer = tf.keras.layers.Dense(dff, activation='relu')
  output_layer = tf.keras.layers.Dense(d_model)

  def feed_forward(x):
    return output_layer(hidden_layer(x))

  return feed_forward


self_attention = get_self_attention(ARGS.d_model, ARGS.num_heads)
feed_forward = get_feed_forward(ARGS.d_model, ARGS.dff)
albert_dynamics = get_albert_dynamics(self_attention, feed_forward)

t0 = tf.constant(0.)

if ARGS.solver == 'rk4':
  solver = RK4Solver(0.01)
elif ARGS.solver == 'rkf56':
  solver = RKF56Solver(0.01, tol=1e-3, min_dt=1e-2)
else:
  raise ValueError(f'Unknown solver: "{ARGS.solver}"')

signature = [[tf.TensorSpec(shape=[None, None, ARGS.d_model], dtype=tf.float32),
              tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
              tf.TensorSpec(shape=[None, None, ARGS.d_model], dtype=tf.float32)]]

node_fn = get_node_function(solver, t0, albert_dynamics, signature=signature)

t1 = tf.constant(1.)
x0 = [tf.random.uniform(shape=[32, 16, ARGS.d_model]),
      tf.zeros(shape=[32, 16]),
      tf.random.uniform(shape=[32, 16, ARGS.d_model])]
x1 = node_fn(t1, x0)
print(f'\nOutput shapes: {[_.shape for _ in x1]}')
