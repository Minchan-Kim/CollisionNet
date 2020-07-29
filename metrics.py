import tensorflow as tf


class FalsePositives(tf.keras.metrics.Metric):
    def __init__(self, thresholds = 0.5, name = 'false_positives', **kwargs):
        super(FalsePositives, self).__init__(name = name, **kwargs)
        self.false_positives = self.add_weight(name = 'fp', initializer = 'zeros')
        self.thresholds = thresholds

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true[:, 0], tf.bool)
        y_pred = tf.math.greater_equal(y_pred[:, 0], self.thresholds)
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.math.multiply(values, sample_weight)
        self.false_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_positives


class FalseNegatives(tf.keras.metrics.Metric):
    def __init__(self, thresholds = 0.5, name = 'false_negatives', **kwargs):
        super(FalseNegatives, self).__init__(name = name, **kwargs)
        self.false_negatives = self.add_weight(name = 'fn', initializer = 'zeros')
        self.thresholds = thresholds

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true[:, 0], tf.bool)
        y_pred = tf.math.greater_equal(y_pred[:, 0], self.thresholds)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.math.multiply(values, sample_weight)
        self.false_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_negatives