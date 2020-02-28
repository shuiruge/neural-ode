import tensorflow as tf
from node.utils.callbacks import LayerInspector


X = tf.constant([[2., 3.]])
y = tf.constant([[1.]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
])
model.compile(loss='mse')

inspector = LayerInspector(samples=(X, y),
                           level='original',
                           skip_step=1)
model.fit(X, y, epochs=2, callbacks=[inspector])
logs = inspector.logs

print(logs)
