#!/usr/bin/env python3
import os
import multiprocessing
import tensorflow as tf
slim = tf.contrib.slim

import nima


tf.app.flags.DEFINE_integer(
    'batch_size', 200, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_epochs', 20, 'The maximum number of training epochs.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'log_dir', None,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None, 'Path to the model checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_boolean(
    'eval', False, 'Whether to run evaluation instead of train.')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10, 'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'cpu_cores', multiprocessing.cpu_count(),
    'The number of CPU cores to use for dataset preprocessing.')

FLAGS = tf.app.flags.FLAGS


def preprocess(example, num_classes=10, is_training=True):
    """Extract and preprocess dataset features.

    Args:
      example: an instance of protobuf-encoded example.
      num_classes: number of predicted classes. Defaults to 10.
      is_training: whether is training or not.

    Returns:
      A tuple of `image` and `scores` tensors.
    """
    features = {'scores': tf.VarLenFeature(tf.float32),
                'image': tf.FixedLenFeature((), tf.string)}
    parsed = tf.parse_single_example(example, features)
    image = tf.image.decode_jpeg(parsed['image'], channels=3)
    image = nima.preprocess_image(image, is_training=is_training)
    scores = parsed['scores']
    scores = tf.sparse_tensor_to_dense(scores)
    scores = tf.reshape(scores, [num_classes])
    scores = scores / tf.reduce_sum(scores, axis=-1, keepdims=True)
    return image, scores


def get_dataset(dataset_dir, split_name, batch_size, workers):
    """Load and preprocess a dataset.

    Args:
      dataset_dir: path to the TFRecord files.
      split_name: train or validation split.
      batch_size: number of items per batch.
      workers: the number of parallel preprocessing workers to run.

    Returns:
      A tuple of Dataset iterator and a number of examples.
    """
    folder = os.path.join(dataset_dir, '{}_*.tfrecord'.format(split_name))
    filenames = tf.data.Dataset.list_files(folder)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess, num_parallel_calls=workers)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)

    filename = '{}.txt'.format(split_name)
    with open(os.path.join(dataset_dir, filename), 'r') as f:
        examples = int(f.read().strip())

    return dataset.make_one_shot_iterator(), examples


def _get_init_fn():
    """Return a function that 'warm-starts' the training.

    Returns:
      An init function.
    """
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from {}'.format(checkpoint_path))

    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        global_step = tf.train.create_global_step()

        dataset_iterator, num_samples = get_dataset(
            FLAGS.dataset_dir, FLAGS.split_name, FLAGS.batch_size,
            FLAGS.cpu_cores)

        images, scores = dataset_iterator.get_next()
        predictions, end_points = nima.get_model(images)

        batches_epoch = num_samples // FLAGS.batch_size
        number_of_steps = batches_epoch * FLAGS.max_epochs

        if FLAGS.eval:
            correlation = slim.metrics.streaming_pearson_correlation(
                predictions, scores)
            metrics, updates = slim.metrics.aggregate_metric_map(
                {'Correlation': correlation})

            for name, value in metrics.items():
                summary_name = 'eval/{}'.format(name)
                op = tf.summary.scalar(summary_name, value, collections=[])
                op = tf.Print(op, [value], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            checkpoint_path = tf.train.latest_checkpoint(FLAGS.log_dir)
            variables_to_restore = slim.get_variables_to_restore()

            tf.logging.info('Evaluating {}'.format(checkpoint_path))
            slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.log_dir,
                num_evals=number_of_steps,
                eval_op=list(updates.values()),
                variables_to_restore=variables_to_restore)
        else:
            loss = nima.emd_loss(scores, predictions, r=2)
            tf.losses.add_loss(loss)

            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate, global_step, batches_epoch,
                FLAGS.learning_rate_decay_factor, staircase=True,
                name='exponential_decay_learning_rate')

            optimizer = tf.train.AdamOptimizer(learning_rate)
            total_loss = tf.losses.get_total_loss()

            summaries = []
            for var in slim.get_model_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))
            summaries.append(
                tf.summary.scalar('training/learning_rate', learning_rate))
            for end_point in end_points:
                var = end_points[end_point]
                summaries.append(
                    tf.summary.histogram('activations/' + end_point, var))
                summaries.append(tf.summary.scalar('sparsity/' + end_point,
                    tf.nn.zero_fraction(var)))
            summaries.append(tf.summary.scalar('total_loss', total_loss))

            summary_op = tf.summary.merge(summaries, name='summary_op')
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            slim.learning.train(
                train_op,
                logdir=FLAGS.log_dir,
                init_fn=_get_init_fn(),
                summary_op=summary_op,
                number_of_steps=number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs)


if  __name__ =='__main__':
    tf.flags.mark_flags_as_required(['dataset_dir', 'log_dir'])
    tf.app.run()
