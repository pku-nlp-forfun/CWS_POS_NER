# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-28 22:24:44
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-31 23:02:38

import constant as con
import numpy as np
import pickle
import tensorflow as tf

from numba import jit
from typing import List
from util import echo, time_str, log


def tf_constant(x: List, y: List, z: List):
    ''' init tf constant params '''
    return tf.constant(np.array(x).astype(np.float32)), tf.constant(y), tf.constant(z)


def score_matrix(x_t, weights, num_seq: int, num_word: int, num_fea: int, num_tag: int):
    x_t_matrix = tf.reshape(x_t, [-1, num_fea])
    matrix_scores = tf.matmul(x_t_matrix, weights)
    scores = tf.reshape(matrix_scores, [num_seq, num_word, num_tag])
    return scores


def mask(num_word: int, seq_len: List):
    mask_matrix = (np.expand_dims(np.arange(num_word), axis=0)
                   < np.expand_dims(seq_len, axis=1))
    total_labels = np.sum(seq_len)

    return mask_matrix, total_labels


def evaluation(y: List, y_predict: List, seq: List, mask, total_label, types: str):
    # print(y.shape, y_predict.shape, mask.shape)
    y = sum([list(jj[:sum(mask[ii])]) for ii, jj in enumerate(y)], [])
    y = [int(ii > 1) for ii in y]
    y_predict = sum([list(jj[:sum(mask[ii])])
                     for ii, jj in enumerate(y_predict)], [])
    y_predict = [int(ii > 1) for ii in y_predict]
    change_idx, idx = [], -1
    for ii in seq:
        change_idx.append(ii + idx)
        idx += ii
    for ii in change_idx:
        y_predict[ii] = 1

    # correct_labels = np.sum((y == y_predict) * mask)
    # accuracy = 100.0 * correct_labels / float(total_label)
    p, r, macro_f1 = fastF1(y, y_predict)
    print(f"{types} P: {p:.2f}%, R: {r:.2f}%, Macro_f1: {macro_f1:.2f}%")

    return p, r, macro_f1


@jit
def fastF1(result, predict):
    ''' cws f1 score calculate '''
    recallNum = sum(result)
    precisionNum = sum(predict)
    last_result, last_predict, trueNum = -1, -1, 0
    for ii in range(len(result)):
        if result[ii] and result[ii] == predict[ii] and last_predict == last_result:
            trueNum += 1
        if result[ii]:
            last_result = ii
        if predict[ii]:
            last_predict = ii
    # print(trueNum, precisionNum, recallNum)
    r = trueNum / recallNum if recallNum else 0
    p = trueNum / precisionNum if precisionNum else 0
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0

    return p * 100, r * 100, macro_f1 * 100


def crf_tf(train_x: List, train_y: List, train_seq: List, train_se: List,
           dev_x: List, dev_y: List, dev_seq: List, dev_se: List,
           test_x: List, test_y: List, test_seq: List, test_se: List, num_tag: int):
    ''' crf base on tensorflow '''

    with tf.Graph().as_default(), tf.Session() as session:

        train_x_init = tf.placeholder(
            tf.float32, shape=np.array(train_x).shape)
        train_x_t = tf.Variable(train_x_init)

        train_y_t = tf.constant(train_y)
        train_seq_t = tf.constant(train_seq)
        dev_x_t, dev_y_t, dev_seq_t = tf_constant(dev_x, dev_y, dev_seq)
        test_x_t, test_y_t, test_seq_t = tf_constant(test_x, test_y, test_seq)

        num_train_seq, num_train_word, num_fea = train_x.shape
        num_dev_seq, num_dev_word, _ = dev_x.shape
        num_test_seq, num_test_word, _ = test_x.shape
        weights = tf.get_variable(
            "weights", [num_fea, num_tag], dtype=tf.float32)

        train_score = score_matrix(
            train_x_t, weights, num_train_seq, num_train_word, num_fea, num_tag)
        dev_score = score_matrix(
            dev_x_t, weights, num_dev_seq, num_dev_word, num_fea, num_tag)
        test_score = score_matrix(
            test_x_t, weights, num_test_seq, num_test_word, num_fea, num_tag)

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            train_score, train_y_t, train_seq_t)
        # softmaxs_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_y_t, logits=train_score)

        train_viterbi_seq, _ = tf.contrib.crf.crf_decode(
            train_score, transition_params, train_seq_t)
        dev_viterbi_seq, _ = tf.contrib.crf.crf_decode(
            dev_score, transition_params, dev_seq_t)
        test_viterbi_seq, _ = tf.contrib.crf.crf_decode(
            test_score, transition_params, test_seq_t)

        ''' loss calculate '''
        loss = tf.reduce_mean(-log_likelihood)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        session.run(tf.global_variables_initializer(),
                    feed_dict={train_x_init: train_x})

        train_mask, train_total = mask(num_train_word, train_seq)
        dev_mask, dev_total = mask(num_dev_word, dev_seq)
        test_mask, test_total = mask(num_test_word, test_seq)
        best_dev_acc = -1

        log(f'------- {time_str()} -------')

        for i in range(1000):
            train_predict, _ = session.run([train_viterbi_seq, train_op])
            if i % 20 == 0:
                dev_predict = session.run([dev_viterbi_seq])[0]
                print(f'------  \033[92m{i} epochs \033[0m -------')
                train_p, train_r, train_macro_f1 = evaluation(
                    train_y, train_predict, train_se, train_mask, train_total, 'Train')
                dev_p, dev_r, dev_macro_f1 = evaluation(
                    dev_y, dev_predict, dev_se, dev_mask, dev_total, 'Dev')
                log(f'{i}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}|')
                if dev_macro_f1 > best_dev_acc:
                    best_dev_acc = dev_macro_f1
                    test_predict = session.run([test_viterbi_seq])[0]
                    pickle.dump(test_predict, open(
                        f"{con.RESULT['CWS']}.pkl", 'wb'))
                    test_p, test_r, test_macro_f1 = evaluation(
                        test_y, test_predict, test_se, test_mask, test_total, 'Test')

        log(f"Best Dev Macro_f1: {best_dev_acc:.2f}%")
        log(f"Best Test P: {test_p:.2f}%, R: {test_r:.2f}%, Macro_f1: {test_macro_f1:.2f}%")

        echo(0, f"Best Dev Macro_f1: {best_dev_acc:.2f}%")
        echo(
            0, f"Best Test P: {test_p:.2f}%, R: {test_r:.2f}%, Macro_f1: {test_macro_f1:.2f}%")
        return test_predict


def test_params(num_seq: int, num_word: int, num_fea: int, num_tag: int):
    x = np.random.rand(num_seq, num_word, num_fea).astype(np.float32)
    y = np.random.randint(num_tag, size=[num_seq, num_word]).astype(np.int32)
    seq_len = np.random.randint(
        max(num_word - 5, 10), num_word, size=[num_seq]).astype(np.int32)
    seq = np.random.randint(0, 5, size=[num_seq]).astype(np.int32)
    return x, y, seq_len, seq


if __name__ == "__main__":
    num_seq = 10
    num_word = 20
    num_fea = 100
    num_tag = 4
    train_x, train_y, train_seq, train_se = test_params(
        num_seq, num_word, num_fea, num_tag)
    dev_x, dev_y, dev_seq, dev_se = test_params(10, num_word, num_fea, num_tag)
    test_x, test_y, test_seq, test_se = test_params(10, 30, num_fea, num_tag)

    crf_tf(train_x, train_y, train_seq, train_se, dev_x, dev_y,
           dev_seq, dev_se, test_x, test_y, test_seq, test_se, num_tag)
