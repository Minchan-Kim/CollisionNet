import tensorflow as tf
import normalization as nr
import numpy as np
import os
import argparse
from pathlib import Path


def write(data, src, dest, time_window, tools, dtype):
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
    filename = dest + '/test{}_'.format(file_index) + description
    writer = tf.io.TFRecordWriter(filename + '.tfrecord')
    for i in range((data.shape)[0]):
        feature = {
            'x': tf.train.Feature(float_list = tf.train.FloatList(value = data[i, :-4])),
            'y': tf.train.Feature(float_list = tf.train.FloatList(value = data[i, -4:-2]))
        }
        example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example_proto.SerializeToString())
    writer.flush()
    writer.close()
    np.savetxt((filename + '.csv'), data[:, -4:], delimiter = ',')


parser = argparse.ArgumentParser()
parser.add_argument('path', nargs = '?', default = '/home/dyros/mc_ws/CollisionNet/data/test')
parser.add_argument('--tools', nargs = '*')
parser.add_argument('--dtype', nargs = '*')
args = parser.parse_args()
dest = args.path
tools = args.tools
dtype = args.dtype

path = Path('/home/dyros/mc_ws/Offline_Experiment')
paths = list(path.glob('**/DRCL_Data.txt*'))
if tools is not None:
    paths = [path for path in paths for tool in tools if tool in str(path)]
if dtype is not None:
    paths = [path for path in paths for dt in dtype if dt in str(path)]

for path in paths:
    nr.Normalize(path, dest, write, 32, 1000, tools, dtype)
    print(str(path))

"""
p1 = Path('/home/dyros/mc_ws/Offline_Experiment/20191122_Test/robot1')
paths = []
dir1 = [x for x in p1.iterdir() if (x.is_dir() and (x.name in Tool_list))]
for p2 in dir1:
    dir2 = [x for x in p2.iterdir() if (x.is_dir() and (x.name in Data_type_list))]
    for p3 in dir2:
        for x in p3.iterdir():
            if x.match('DRCL_Data.txt*'):
                paths.append(x)
"""