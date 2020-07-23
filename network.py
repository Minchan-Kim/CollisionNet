import tensorflow as tf


class CollisionNet:
    def __init__(self, num_data, time_window, learning_rate, batch_size = None, minibatch_size = None):
        self.model = tf.keras.models.Sequential([
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
            # 32 window
            tf.keras.layers.Conv1D(512, 2, padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(512, 2, padding = 'valid', dilation_rate = 16),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # flatten
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation = 'softmax')
        ])
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.metrics = [
            tf.keras.metrics.Mean(name = 'loss'),
            tf.keras.metrics.CategoricalAccuracy(name = 'acc'),
            tf.keras.metrics.CategoricalCrossentropy(name = 'val_loss'),
            tf.keras.metrics.CategoricalAccuracy(name = 'val_acc')
        ]
        self.steps = None
        if (batch_size is not None) and (minibatch_size is not None):
            self.batch_gradients = []
            for variable in self.model.trainable_variables:
                self.batch_gradients.append(tf.Variable(tf.zeros_like(variable)))
            self.num_variables = len(batch_gradients)
            self.steps = int(batch_size / minibatch_size)
            self.step_count = 0

    @tf.function
    def train(self, dataset):
        for x, y in dataset:
            with tf.GradientTape() as tape:
                y_pred = self.model(x, training = True)
                loss = self.loss(y_true = y, y_pred = y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.metrics[0].update_state(loss)
            self.metrics[1].update_state(y, y_pred)

    @tf.function
    def train_using_gradient_accumulation(self, dataset):
        for x, y in dataset:
            with tf.GradientTape() as tape:
                y_pred = self.model(x, training = True)
                loss = self.loss(y_true = y, y_pred = y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            for i in range(self.num_variables):
                self.batch_gradients[i].assign_add(gradients[i] / steps)
            self.step_count += 1
            if self.step_count >= steps:
                self.optimizer.apply_gradients(zip(self.batch_gradients, self.model.trainable_variables))
                self.step_count = 0
                for i in range(num_variables):
                    self.batch_gradients[i].assign(tf.zeros_like(self.batch_gradients[i]))
            self.metrics[0].update_state(loss)
            self.metrics[1].update_state(y, y_pred)

    @tf.function
    def evaluate(self, validation_dataset):
        for x, y in validation_dataset:
            y_pred = self.model(x)
            self.metrics[2].update_state(y, y_pred)
            self.metrics[3].update_state(y, y_pred)

    def fit(self, x, epochs, callbacks = None, validation_data = None):
        if self.steps is not None:
            train = self.train_using_gradient_accumulation
        else:
            train = self.train
        for epoch in range(epochs):
            train(x)
            if validation_data is not None:
                self.evaluate(validation_data)
            print('Epoch {}/{}'.format((epoch + 1), epochs))
            result = ''
            for metric in self.metrics:
                result += '{}: {:.4} - '.format(metric.name, metric.result())
            print(result[:-3])
            if callbacks is not None:
                logs = dict([(metric.name, metric.result()) for metric in self.metrics])
                for callback in callbacks:
                    callback(epoch, logs)
            for metric in self.metrics:
                metric.reset_states()

    def save(self, filepath, overwrite = True, include_optimizer = True, save_format = None):
        self.model.save(filepath, overwrite, include_optimizer, save_format)

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model


if __name__ == "__main__":
    model = CollisionNet(49, 16, 0.001)
    model.summary()