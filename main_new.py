import tensorflow as tf
import dataset as ds
import time


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def CollisionNet(num_data, time_window):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape = (time_window, num_data)),
        tf.keras.layers.Conv1D(128, 2, padding = 'valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(128, 2, padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(128, 2, padding = 'valid', dilation_rate = 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(128, 2, padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(256, 2, padding = 'valid', dilation_rate = 4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(256, 2, padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(512, 2, padding = 'valid', dilation_rate = 8),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(512, 2, padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(512, 2, padding = 'valid', dilation_rate = 16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation = 'softmax')
    ])
    return model

model = CollisionNet(49, 32)
model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = 'categorical_crossentropy',
        metrics = ['acc']
    )

dataset = ds.Dataset('/home/dyros/mc_ws/CollisionNet/data_0_00kg/training_new', 49, 32, 20000, 100, pattern = '*.tfrecord', processed = True)

loss_object = tf.keras.losses.CategoricalCrossentropy()
acc_tracker = tf.keras.metrics.CategoricalAccuracy()

batch_gradients = []
for variable in model.trainable_variables:
    batch_gradients.append(tf.Variable(tf.zeros_like(variable)))
num_variables = len(batch_gradients)
steps = 10
step_count = None
"""
@tf.function
def train():
    step_count = 0
    for x, y in dataset:
        with tf.GradientTape() as t:
            y_pred = model(x, training = True)
            loss = loss_object(y_true = y, y_pred = y_pred)
        gradients = t.gradient(loss, model.trainable_variables)
        for i in range(num_variables):
            batch_gradients[i].assign_add(gradients[i] / steps)
        step_count += 1
        if step_count == steps:
            model.optimizer.apply_gradients(zip(batch_gradients, model.trainable_variables))
            step_count = 0
            for i in range(num_variables):
                batch_gradients[i].assign(tf.zeros_like(batch_gradients[i]))
        acc_tracker.update_state(y, y_pred)
"""

@tf.function
def train():
    for x, y in dataset:
        with tf.GradientTape() as t:
            y_pred = model(x, training = True)
            loss = loss_object(y_true = y, y_pred = y_pred)
        gradients = t.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc_tracker.update_state(y, y_pred)

validation_dataset = ds.Dataset(
    '/home/dyros/mc_ws/CollisionNet/data/32/validation', 49, 32, 0, 100, pattern = '*0_00kg.tfrecord', num_parallel_calls = 3, processed = False, drop_remainder = False)

for i in range(1):
    print("Epoch {}".format(i + 1))
    start_time = time.time()
    train()
    elapsed_time = time.time() - start_time
    print(acc_tracker.result().numpy())
    loss, acc = model.evaluate(validation_dataset, verbose = 0)
    print(acc)
    print(elapsed_time)