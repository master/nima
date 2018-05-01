"""Tests for the NIMA."""

import tensorflow as tf
import numpy as np

import nima


class NimaTest(tf.test.TestCase):

    def test_tril_indices(self):
        with self.test_session() as sess:
            indices_3 = sess.run(nima.tril_indices(3))
            indices_2_off = sess.run(nima.tril_indices(2, 1))
        self.assertAllClose(indices_3[0],
                            np.array([0, 1, 1, 2, 2, 2], dtype=np.int32))
        self.assertAllClose(indices_3[1],
                            np.array([0, 0, 1, 0, 1, 2], dtype=np.int32))
        self.assertAllClose(indices_2_off[0],
                            np.array([0, 0, 1, 1], dtype=np.int32))
        self.assertAllClose(indices_2_off[1],
                            np.array([0, 1, 0, 1], dtype=np.int32))

    def test_ecdf(self):
        with self.test_session() as sess:
            ecdf = sess.run(nima.ecdf(
                tf.constant([[1, 1, 1, 2], [1, 2, 3, 4]], dtype=tf.float32)))
        self.assertAllClose(ecdf, np.array(
            [[1., 2., 3., 5.], [1., 3., 6., 10.]], dtype=np.float32))

    def test_emd_loss(self):
        p1 = tf.constant([[0.1, 0.2, 0.3, 0.5], [0.3, 0.3, 0.3, 0.1]])
        p2 = tf.constant([[0.3, 0.3, 0.3, 0.1], [0.3, 0.3, 0.4, 0.0]])
        with self.test_session() as sess:
            loss = sess.run(nima.emd_loss(p1, p2))
        self.assertAllClose(loss, 0.14489579)

    def test_scores_stats(self):
        scores = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0],
                              [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.1, 0.2, 0.3, 0]])
        scores = scores / tf.reduce_sum(scores, axis=-1, keepdims=True)
        with self.test_session() as sess:
            means, stds = sess.run(nima.scores_stats(scores))
        self.assertAllClose(means, [6.3333333, 5.3])
        self.assertAllClose(stds, [2.2110815, 2.05182862])
