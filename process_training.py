import tensorflow as tf
import numpy as np
import normalization as nr
from collections import deque
from pathlib import Path
import argparse


def write(data, src, dest, time_window, tools, dtype):
    windows = []
    window = deque(maxlen = time_window)
    for i in range(time_window - 1):
        window.append(data[i, :-4])
    for i in range((time_window - 1), (data.shape)[0]):
        window.append(data[i, :-4])
        x = np.vstack(tuple(window))
        y = data[i, -4:-2]
        windows.append((x, y))
    np.random.shuffle(windows)
    file_index = len([item for item in Path(dest).iterdir() if (item.is_file() and item.suffix == '.tfrecord')]) + 1
    description = ''
    src = str(src)
    if 'collision' in src:
        description += 'collision_'
    else:
        description += 'free_'
    start = src.index('kg') - 4
    end = start + 6
    description += src[start:end]
    filename = dest + '/training{}_'.format(file_index) + description
    writer = tf.io.TFRecordWriter(filename  + '.tfrecord')
    for x, y in windows:
        x = [tf.io.serialize_tensor(x.astype(np.float32)).numpy()]
        feature = {
            'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = x)),
            'y': tf.train.Feature(float_list = tf.train.FloatList(value = y))
        }
        example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example_proto.SerializeToString())
    writer.flush()
    writer.close()


parser = argparse.ArgumentParser()
parser.add_argument('path', nargs = '?', default = '/home/dyros/mc_ws/CollisionNet/data/32/training')
parser.add_argument('--time_window', type = int, default = 32)
parser.add_argument('--tools', nargs = '*')
parser.add_argument('--dtype', nargs = '*')
args = parser.parse_args()
dest = args.path
time_window = args.time_window
tools = args.tools
dtype = args.dtype

tf.get_logger().setLevel('WARN')

path = Path('/home/dyros/mc_ws/data')
paths = list(path.glob('robot*/**/Reduced_DRCL_Data.txt'))
if tools is not None:
    paths = [path for path in paths for tool in tools if tool in str(path)]
if dtype is not None:
    paths = [path for path in paths for dt in dtype if dt in str(path)]

for path in paths:
    nr.Normalize(path, dest, write, time_window, 100, tools, dtype)
    print(str(path))