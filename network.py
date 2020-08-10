import tensorflow as tf


class SequentialWithGradAccum(tf.keras.models.Sequential):
    def __init__(self, batch_size, minibatch_size):
        super(SequentialWithGradAccum, self).__init__()
        self.steps = tf.constant(int(batch_size / minibatch_size), dtype = tf.float32)
        self.step_count = tf.Variable(0, trainable = False, dtype = tf.float32)
        
    def initialize_batch_gradients(self):
        self.batch_gradients = []
        for variable in self.trainable_variables:
            self.batch_gradients.append(tf.Variable(tf.zeros_like(variable), trainable = False))
        self.num_variables = len(self.batch_gradients)

    def assign_gradients(self, *gradients):
        for i in range(self.num_variables):
            self.batch_gradients[i].assign(gradients[i])

    def accumulate_gradients(self, *gradients):
        for i in range(self.num_variables):
            self.batch_gradients[i].assign_add(gradients[i])

    def average_gradients(self):
        for i in range(self.num_variables):
            self.batch_gradients[i].assign(self.batch_gradients[i] / self.steps)

    def train_step(self, dataset):
        x, y = dataset
        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
            loss = self.compiled_loss(y, y_pred, regularization_losses = self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        if tf.math.equal(self.step_count, 0):
            tf.py_function(self.assign_gradients, gradients, [])
        else:
            tf.py_function(self.accumulate_gradients, gradients, [])
        self.step_count.assign_add(1)
        if tf.math.greater_equal(self.step_count, self.steps):
            tf.py_function(self.average_gradients, [], [])
            self.optimizer.apply_gradients(zip(self.batch_gradients, self.trainable_variables))
            self.step_count.assign(0)
        self.compiled_metrics.update_state(y, y_pred)

        return {metric.name: metric.result() for metric in self.metrics}


def CollisionNet(num_data, time_window, **kwargs):
    batch_size = kwargs.get('batch_size')
    minibatch_size = kwargs.get('minibatch_size')
    if ((batch_size is not None) and (minibatch_size is not None)):
        model = SequentialWithGradAccum(batch_size, minibatch_size)
    else:
        model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = (time_window, num_data)))
    model.add(tf.keras.layers.Conv1D(128, 3, padding = 'valid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(128, 3, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(128, 3, padding = 'valid', dilation_rate = 2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(128, 3, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(256, 3, padding = 'valid', dilation_rate = 4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(256, 3, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(512, 3, padding = 'valid', dilation_rate = 8))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Conv1D(512, 2, padding = 'same'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Conv1D(512, 2, padding = 'valid', dilation_rate = 16))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    if kwargs.get('compile', True):
        learning_rate = kwargs.get('learning_rate', 0.001)
        beta_1 = kwargs.get('beta_1', 0.9)
        beta_2 = kwargs.get('beta_2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-7)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon),
            loss = 'categorical_crossentropy',
            metrics = ['acc']
        )
    if (batch_size is not None) and (minibatch_size is not None):
        model.initialize_batch_gradients()

    return model


if __name__ == '__main__':
    model = CollisionNet(49, 32, 0.0001, 1000, 100)
    model.summary()