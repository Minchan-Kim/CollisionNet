import tensorflow as tf
import pandas as pd
from pathlib import Path
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('path', nargs = '?', default = '/home/dyros/mc_ws/CollisionNet/data/32/validation')
args = parser.parse_args()
dest = args.path
paths = Path(dest).glob('*temp*.csv')
for path in paths:
    data = pd.read_csv(path, header = None).to_numpy()
    size = (data.shape)[0]
    filename = dest + '/validation{}.tfrecord'.format(path.name[4:-4])
    writer = tf.io.TFRecordWriter(filename)
    for i in range(size):
        feature = {
            'x': tf.train.Feature(float_list = tf.train.FloatList(value = data[i, :-2])),
            'y': tf.train.Feature(float_list = tf.train.FloatList(value = data[i, -2:]))
        }
        example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example_proto.SerializeToString())
    writer.flush()
    writer.close()
    print(str(path))
    os.remove(path)