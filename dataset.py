import tensorflow as tf
import os


_num_data = 1
_num_time_window = 1

def parse1(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature(shape = (_num_data,), dtype = tf.float32),
        'y': tf.io.FixedLenFeature(shape = (2,), dtype = tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def parse2(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature(shape = [], dtype = tf.string),
        'y': tf.io.FixedLenFeature(shape = (2,), dtype = tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)    

def process1(filename):
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse1).prefetch(1)

    x_dataset = dataset.map(lambda data: data['x'])
    x_dataset = x_dataset.window(_num_time_window, shift = 1, stride = 1)
    x_dataset = x_dataset.flat_map(lambda window: window.batch(_num_time_window))

    y_dataset = dataset.map(lambda data: data['y'])
    y_dataset = y_dataset.skip(_num_time_window - 1)

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    return dataset

def process2(filename):
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse2).prefetch(1)

    x_dataset = dataset.map(lambda data: data['x'])
    x_dataset = x_dataset.map(lambda x: tf.io.parse_tensor(x, out_type = tf.float32))
    x_dataset = x_dataset.map(lambda x: tf.ensure_shape(x, (_num_time_window, _num_data)))

    y_dataset = dataset.map(lambda data: data['y'])

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    return dataset

def OrderedDataset(filenames, num_data, time_window, batch_size, processed = False):
    global _num_data, _num_time_window
    _num_data = num_data
    _num_time_window = time_window

    if processed:
        process = process2
    else:
        process = process1

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(process)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

def Dataset(path, num_data, time_window, buffer_size, batch_size, pattern = '*.tfrecord', cycle_length = 1, num_parallel_calls = None, processed = True, drop_remainder = True):
    global _num_data, _num_time_window
    _num_data = num_data
    _num_time_window = time_window

    if num_parallel_calls is None:
        num_parallel_calls = os.cpu_count()
    cycle_length = max([cycle_length, num_parallel_calls])

    if processed:
        process = process2
    else:
        process = process1    

    dataset = tf.data.Dataset.list_files(file_pattern = (path + '/' + pattern))
    dataset = dataset.interleave(process, cycle_length = cycle_length, num_parallel_calls = num_parallel_calls, deterministic = False)
    if buffer_size > 0:
        dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)
    dataset = dataset.prefetch(1)

    return dataset


if __name__ == "__main__":
    dataset = Dataset('/home/dyros/mc_ws/CollisionNet/data/32/validation_new', 49, 32, 0, 1000, pattern = '*.tfrecord', num_parallel_calls = 1, processed = False, drop_remainder = False)
    #dataset = dataset.take(1)
    
    cnt = 0
    for x, y in dataset:
        #print(x.numpy())
        #print(y.numpy())
        #print(x[0])
        #print(y[0])
        #print(x.shape)
        #print(y.shape)
        cnt += (x.shape[0])
    print(cnt)