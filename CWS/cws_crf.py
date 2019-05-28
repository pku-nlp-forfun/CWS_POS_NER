# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-28 22:24:44
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-28 23:04:13

import numpy as np
import tensorflow as tf
from typing import List


def crf_tf(input_matrix: List, label: List, seq_len: List,
           num_x: int, num_y: int, num_features: int, num_tag: int):
    ''' crf base on tensorflow '''

    with tf.Graph().as_default(), tf.Session() as session:

        input_t = tf.constant(input_matrix)
        label_t = tf.constant(label)
        seq_len_t = tf.constant(seq_len)

        # Compute unary scores from a linear layer.
        weights = tf.get_variable("weights", [num_features, num_tags])
        input_t_matrix = tf.reshape(input_t, [-1, num_features])
        matrix_unary_scores = tf.matmul(input_t_matrix, weights)
        unary_scores = tf.reshape(matrix_unary_scores, [
                                  num_examples, num_words, num_tags])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            unary_scores, label_t, seq_len_t)

        # Compute the viterbi sequence and score.
        viterbi_seq, viterbi_score = tf.contrib.crf.crf_decode(
            unary_scores, transition_params, seq_len_t)

        # Add a training op to tune the parameters.
        loss = tf.reduce_mean(-log_likelihood)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        session.run(tf.global_variables_initializer())

        mask = (np.expand_dims(np.arange(num_words), axis=0) <
                np.expand_dims(seq_len, axis=1))
        total_labels = np.sum(seq_len)

        for i in range(2000):
            viterbi_seq_t, _ = session.run([viterbi_seq, train_op])
            if i % 100 == 0:
                correct_labels = np.sum((y == viterbi_seq_t) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)


if __name__ == "__main__":
    num_examples = 10
    num_words = 20
    num_features = 100
    num_tags = 5
    
    x = np.random.rand(num_examples, num_words,
                       num_features).astype(np.float32)
    y = np.random.randint(
        num_tags, size=[num_examples, num_words]).astype(np.int32)
    sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

    crf_tf(x, y, sequence_lengths, num_examples, num_words, num_features, num_tags)

