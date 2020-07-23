import tensorflow as tf


def CollisionNet(num_data, time_window, learning_rate):
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
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = 'categorical_crossentropy',
        metrics = ['acc']
    )
    return model


if __name__ == "__main__":
    model = CollisionNet(66, 32, 0.0002)
    model.summary()