"""
Use ALBERT dynamics to do time series forecasting.

The dataset and baseline model is
[here](https://tensorflow.google.cn/tutorials/structured_data/time_series).

Herein, we will not use the positional encoding. Instead, we add features
"month", "day", and "hour" for indicating the time (i.e. sequential position).
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from node.core import get_node_function
from node.solvers import runge_kutta as rk
from node.applications.nlp.albert import get_albert_dynamics
from node.applications.nlp.utils import PositionalEncoding, SelfAttention


class EmbeddingLayer(tf.keras.layers.Layer):

  def __init__(self, depth, max_sequence_length=None):
    super().__init__()
    self.depth = depth
    self.max_sequence_length = max_sequence_length

    self._embedding = tf.keras.layers.Dense(depth)
    if max_sequence_length is None:
      self._positional_encoding = None
    else:
      self._positional_encoding = (
          PositionalEncoding(depth, max_sequence_length))

  def call(self, x):
    x = self._embedding(x)
    if self._positional_encoding is not None:
      x = self._positional_encoding(x)
    return x


class EncodingLayer(tf.keras.layers.Layer):
  r"""
  Examples:

  ```python
  from node.solvers.runge_kutta import RKF56Solver

  depth = 8
  num_heads = 4
  dff = 128
  solver = RKF56Solver(0.01, 1e-3, 1e-2)
  encoding_layer = EncodingLayer(depth, num_heads, dff, solver, 1.)

  x = tf.random.uniform(shape=[16, 32, depth])
  mask = tf.zeros(shape=[16, 1, 32, 1])

  y = encoding_layer([x, mask])
  print([_.shape for _ in y])
  ```
  """

  def __init__(self, depth, num_heads, dff, solver, t):
    super().__init__()

    _self_attention = SelfAttention(depth, num_heads)

    def self_attention(x, mask):
      return _self_attention([x, mask])

    hidden_layer = tf.keras.layers.Dense(dff, activation='relu')
    output_layer = tf.keras.layers.Dense(depth)

    def feed_forward(x):
      return output_layer(hidden_layer(x))

    dynamics = get_albert_dynamics(self_attention, feed_forward)
    t0 = tf.constant(0.)
    t = tf.convert_to_tensor(t)
    signature = [[
        tf.TensorSpec(shape=[None, None, depth], dtype=self.dtype),
        tf.TensorSpec(shape=[None, 1, None, 1], dtype=self.dtype),
        tf.TensorSpec(shape=[None, None, depth], dtype=self.dtype)
    ]]
    node_fn = get_node_function(solver, t0, dynamics, signature=signature)

    def output_fn(x):
      return node_fn(t, x)[0]

    self._self_attention = self_attention
    self._output_fn = output_fn

  def call(self, inputs):
    x, mask = inputs
    att = self._self_attention(x, mask)
    return self._output_fn([x, mask, att])


class AlbertModel(tf.keras.Model):

  def __init__(self, depth, num_heads, dff, output_dim,
               max_sequence_length=None,
               solver='rkf56', t=1.):
    super().__init__()

    if solver == 'rk4':
      solver = rk.RK4Solver(dt=1e-2)
    elif solver == 'rkf56':
      solver = rk.RKF56Solver(dt=1e-2, tol=1e-2, min_dt=1e-2)
    else:
      pass

    self._solver = solver
    self._embedding_layer = EmbeddingLayer(depth, max_sequence_length)
    self._encoding_layer = EncodingLayer(depth, num_heads, dff, solver, t)

    self._output_layer = tf.keras.layers.Dense(output_dim)

  def call(self, inputs):
    x, mask = inputs
    x = self._embedding_layer(x)
    encoded = self._encoding_layer([x, mask])
    y = self._output_layer(encoded)
    return y


def multivariate_data(dataset, target, start_index, end_index, size, step):
  data = []
  labels = []

  if end_index is None:
    end_index = len(dataset) - size

  for i in range(start_index, end_index):
    indices = range(i, i + size, step)
    data.append(dataset[indices])
    labels.append(target[indices])

  return np.array(data), np.array(labels)[:,:,np.newaxis]


def load_data():
  zip_path = tf.keras.utils.get_file(
      origin=('https://storage.googleapis.com/tensorflow/'
              'tf-keras-datasets/jena_climate_2009_2016.csv.zip'),
      fname='jena_climate_2009_2016.csv.zip',
      extract=True)
  csv_path, _ = os.path.splitext(zip_path)
  df = pd.read_csv(csv_path)

  def parse_datetime(dt_str):
    return datetime.strptime(dt_str, '%d.%m.%Y %H:%M:%S')

  df['Date Time'] = df['Date Time'].apply(parse_datetime)
  df['year'] = df['Date Time'].apply(lambda x: x.year)
  df['month'] = df['Date Time'].apply(lambda x: x.month)
  df['day'] = df['Date Time'].apply(lambda x: x.day)
  df['hour'] = df['Date Time'].apply(lambda x: x.hour)
  df['minute'] = df['Date Time'].apply(lambda x: x.minute)

  return df

def get_training_and_test_features(data):
  input_features = ['year', 'month', 'day', 'hour', 'minute',
                    'p (mbar)', 'rho (g/m**3)']
  target_feature = ['T (degC)']
  features = data[input_features + target_feature]
  features.index = data['Date Time']

  dataset = features.values
  train_split = int(0.8 * len(dataset))

  data_mean = dataset[:train_split].mean(axis=0)
  data_std = dataset[:train_split].std(axis=0)
  dataset = (dataset - data_mean) / data_std

  size = 1 * 24 * 6
  step = 6
  x_train, y_train = multivariate_data(
      dataset[:, :-1], dataset[:, -1], 0, train_split, size, step)
  x_test, y_test = multivariate_data(
      dataset[:, :-1], dataset[:, -1], train_split, None, size, step)
  return (x_train, y_train, x_test, y_test)


data = load_data()
x_train, y_train, x_test, y_test = get_training_and_test_features(data)
print(x_train.shape, y_train.shape)
depth = 32
num_train_data, sequence_length = x_train.shape[:2]
mask_train = np.zeros(shape=[num_train_data, 1, sequence_length, 1])

model = AlbertModel(depth, num_heads=4, dff=128, output_dim=1, t=1.,
                    solver=rk.RK4Solver(0.1))
model.compile(loss='mae')

num_test_data = x_test.shape[0]
mask_test = np.zeros(shape=[num_test_data, 1, sequence_length, 1])

model.fit([x_train, mask_train], y_train, epochs=10,
          validation_data=([x_test, mask_test], y_test))

"""
Epoch 1/10
336440/336440 [==============================] - 2668s 8ms/sample - loss: 0.0401 - val_loss: 0.0359
Epoch 2/10
336440/336440 [==============================] - 2677s 8ms/sample - loss: 0.0263 - val_loss: 0.0391
Epoch 3/10
336440/336440 [==============================] - 2226s 7ms/sample - loss: 0.0240 - val_loss: 0.0315
"""
