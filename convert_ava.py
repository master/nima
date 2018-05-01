#!/usr/bin/env python3
"""Script for converting AVA dataset."""
import os
import random

import tensorflow as tf


tf.app.flags.DEFINE_string(
    'ava_dir', None, 'The directory where the AVA dataset is stored.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the TFRecord files will be stored.')

tf.app.flags.DEFINE_integer(
    'shard_num', 10000, 'Number of examples per TFRecord shard.')

tf.app.flags.DEFINE_float(
    'validation_split', 0.2, 'Fraction of dataset for validation.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    with open(os.path.join(FLAGS.ava_dir, 'AVA.txt'), 'r') as f:
        ava = [line.strip().split() for line in f.readlines()]

    image_path = tf.placeholder(dtype=tf.string)
    jpeg = tf.read_file(image_path)
    decoded = tf.image.decode_jpeg(jpeg, channels=3)

    counts = {'train': 0, 'validation': 0}
    writers = {}

    os.makedirs(FLAGS.dataset_dir)

    with tf.Session() as sess:
        for item in ava:
            filename = os.path.join(FLAGS.ava_dir, 'images', item[1]) + '.jpg'
            try:
                image_data, _ = sess.run(
                    [jpeg, decoded], feed_dict={image_path: filename})

                if random.random() > FLAGS.validation_split:
                    split = 'train'
                else:
                    split = 'validation'

                if split not in writers or counts[split] % FLAGS.shard_num == 0:
                    writer_path = os.path.join(
                        FLAGS.dataset_dir, '{}_{}-{}.tfrecord'.format(
                            split, counts[split],
                            counts[split] + FLAGS.shard_num - 1))
                    writers[split] = tf.python_io.TFRecordWriter(writer_path)

                scores = tf.train.FloatList(value=list(map(int, item[2:12])))
                image = tf.train.BytesList(value=[image_data])
                features = tf.train.Features(feature={
                    'scores': tf.train.Feature(float_list=scores),
                    'image': tf.train.Feature(bytes_list=image)})
                example = tf.train.Example(features=features)
                writers[split].write(example.SerializeToString())
                counts[split] += 1
            except:
                print('Error decoding image: {}'.format(filename))

    for split, count in counts.items():
        filename = '{}.txt'.format(split)
        with open(os.path.join(FLAGS.dataset_dir, filename), 'w') as f:
            f.write('{}\n'.format(count))


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['dataset_dir', 'ava_dir'])
    tf.app.run()
