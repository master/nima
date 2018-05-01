"""Contains the definition for the NIMA network."""
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

IMAGE_SIZE = inception.inception_v2.default_image_size


def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n, m) array.

    Works similarly to `np.tril_indices`

    Args:
      n: the row dimension of the arrays for which the returned indices will
        be valid.
      k: optional diagonal offset (see `np.tril` for details).

    Returns:
      inds: The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = (m1 - m2) >= -k
    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
    return ix1, ix2


def ecdf(p):
    """Estimate the cumulative distribution function.

    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
    function with jump 1/n at each observation (possibly with multiple jumps
    at one place if there are ties).

    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
    observations less or equal to t, i.e.,

    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).

    Args:
      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
        Classes are assumed to be ordered.

    Returns:
      A 2-D `Tensor` of estimated ECDFs.
    """
    n = p.get_shape().as_list()[1]
    indices = tril_indices(n)
    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
    ones = tf.ones([n * (n + 1) / 2])
    triang = tf.scatter_nd(indices, ones, [n, n])
    return tf.matmul(p, triang)


def emd_loss(p, p_hat, r=2, scope=None):
    """Compute the Earth Mover's Distance loss.

    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).

    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.

    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i

    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
        ecdf_p = ecdf(p)
        ecdf_p_hat = ecdf(p_hat)
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
        emd = tf.pow(emd, 1 / r)
        return tf.reduce_mean(emd)


def preprocess_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE,
                     rescale_height=256, rescale_width=256,
                     central_fraction=0.875, is_training=True, scope=None):
    """Pre-process a batch of images for training or evaluation.

    Args:
      image: a tensor of shape [height, width, channels] with the image.
      height: optional Integer, image expected height.
      width: optional Integer, image expected width.
      rescale_height: optional Integer, rescaling height before cropping.
      rescale_width: optional Integer, rescaling width before cropping.
      central_fraction: optional Float, fraction of the image to crop.
      is_training: if true it would transform an image for training,
        otherwise it would transform it for evaluation.
      scope: optional name scope.

    Returns:
      3-D float Tensor containing a preprocessed image.
    """

    with tf.name_scope(scope, 'prep_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if is_training:
            image = tf.image.resize_images(
                image, [rescale_height, rescale_width])
            image = tf.random_crop(image, size=(height, width, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image, [height, width])

        tf.summary.image('final_sampled_image', tf.expand_dims(image, 0))
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def scores_stats(scores):
    """Compute score statistics.

    Args:
      scores: a tensor of shape [batch_size, 10].

    Returns:
      A tuple of 1-D `mean` and `std` `Tensors` with shapes [batch_size].
    """
    values = tf.to_float(tf.range(1, 11))
    values = tf.expand_dims(values, axis=0)
    mean = tf.reduce_sum(values * scores, axis=-1)
    var = tf.reduce_sum(tf.square(values) * scores, axis=-1) - tf.square(mean)
    std = tf.sqrt(var)
    return mean, std


def get_model(images, num_classes=10, is_training=True, weight_decay=4e-5,
              dropout_keep_prob=0.75):
    """Neural Image Assessment from https://arxiv.org/abs/1709.05424

    Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural Image Assessment."
    arXiv preprint arXiv:1709.05424 (2017).

    Args:
      images: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes. Defaults to 10.
      is_training: whether is training or not.
      weight_decay: the weight decay to use for regularizing the model.
      dropout_keep_prob: the percentage of activation values that are retained.
        Defaults to 0.75

    Returns:
      predictions: a tensor of size [batch_size, num_classes].
      end_points: a dictionary from components of the network.
    """
    arg_scope = inception.inception_v2_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits, end_points = inception.inception_v2(
            images, num_classes, is_training=is_training,
            dropout_keep_prob=dropout_keep_prob)

    predictions = tf.nn.softmax(logits)

    return predictions, end_points


