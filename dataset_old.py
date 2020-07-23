import tensorflow as tf
from pathlib import Path


_num_data = 1
_num_time_window = 1

def parse(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature(shape = (_num_data,), dtype = tf.float32),
        'y': tf.io.FixedLenFeature(shape = (2,), dtype = tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def bind(window):
    return window.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(_num_data).batch(_num_time_window)

def get_one_dataset(filename, num_data, time_window):
    global _num_data, _num_time_window
    _num_data = num_data
    _num_time_window = time_window
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse)

    x_dataset = dataset.map(lambda data: data['x'])
    x_dataset = x_dataset.window(time_window, shift = 1, stride = 1)
    x_dataset = x_dataset.flat_map(bind)

    y_dataset = dataset.map(lambda data: data['y'])
    y_dataset = y_dataset.skip(time_window - 1)

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    return dataset

def Dataset(path, num_data, time_window, buffer_size, batch_size):
    p = Path(path)
    records = [record for record in p.iterdir() if record.match('*.tfrecord')]
    datasets = []
    for record in records:
        datasets.append(get_one_dataset(str(record), num_data, time_window))
    dataset = datasets[0]
    for i in range(1, len(datasets)):
        dataset = dataset.concatenate(datasets[i])
    if buffer_size > 0:
        dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset


if __name__ == "__main__":
    dataset = Dataset('/home/dyros/mc_ws/CollisionNet/data/validation', 49, 32, 0, 1)
    #dataset = dataset.take(2)
    sess = tf.Session()
    train_iter = dataset.make_initializable_iterator()
    train_x, train_y = train_iter.get_next()
    sess.run(train_iter.initializer)

    cnt = 0
    while True:
        try:
            x, y = sess.run([train_x, train_y])
            #print(x[0])
            #print(y[0])
            #print(x.shape)
            #print(y.shape)
            cnt += 1
        except tf.errors.OutOfRangeError:
            break
    print(cnt)