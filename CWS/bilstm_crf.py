import constant as con
import numpy as np
import pickle
import tensorflow as tf
import time

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


def evaluation(y: List, y_predict: List, seq: List, types: str):
    # print(np.array(y).shape, np.array(y_predict).shape, seq.shape)
    y = sum([list(jj[:seq[ii]]) for ii, jj in enumerate(y)], [])
    y = [int(ii > 1) for ii in y]
    y_predict = sum([list(jj[:seq[ii]]) for ii, jj in enumerate(y_predict)], [])
    y_predict = [int(ii > 1) for ii in y_predict]
    change_idx, idx = [], -1
    for ii in seq:
        change_idx.append(ii + idx)
        idx += ii
    for ii in change_idx:
        try:
            y_predict[ii] = 1
        except:
            echo(0, ii, len(y_predict))

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


class BiLSTM_CRF_Model():
    ''' bi lstm crf model '''

    def __init__(self, max_len=200, vocab_size=None, num_tag=None, model_save_path=None, embed_size=256, hs=512):
        self.timestep_size = self.max_len = max_len
        self.vocab_size = vocab_size
        self.input_size = self.embedding_size = embed_size
        self.num_tag = num_tag
        self.hidden_size = hs
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.model_save_path = model_save_path

        # Embedding vector
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding", [vocab_size, self.embedding_size], dtype=tf.float32)
        self.train()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_lstm(self, X_inputs):
        # The actual input parameters and the converted output as follows:
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        self.inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        # The input sentence is still padding filled data.
        # Calculate the actual length of each sentence, that is, the actual length of the non-zero non-padding portion.
        self.length = tf.reduce_sum(tf.sign(X_inputs), 1)
        self.length = tf.cast(self.length, tf.int32)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell(), self.lstm_cell(), self.inputs,
                                                                    sequence_length=self.length, dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, [-1, self.hidden_size * 2])
        return output

    def train(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')

        bilstm_output = self.bi_lstm(self.X_inputs)

        echo(1, 'The shape of BiLSTM Layer output:', bilstm_output.shape)

        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size * 2, self.num_tag])
            softmax_b = self.bias_variable([self.num_tag])
            self.y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

            self.scores = tf.reshape(self.y_pred, [-1, self.timestep_size,
                                                   self.num_tag])  # [batchsize, timesteps, num_class]
            print('The shape of Output Layer:', self.scores.shape)
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.y_inputs, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

class BiLSTMTrain(object):
    ''' bi lstm train '''
     
    def __init__(self, data_train:List, data_dev:List, data_test:List, model:BiLSTM_CRF_Model):
        self.data_train = data_train
        self.data_dev = data_dev
        self.data_test = data_test
        self.model = model

    def train(self, max_epoch:int, max_max_epoch:int, tr_batch_size:int, display_num:int=5):
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        echo(0, 'Train shape', self.data_train[0].shape, self.data_train[1].shape)
        echo(0, 'Dev shape', self.data_dev[0].shape, self.data_dev[1].shape)
        echo(0, 'Test shape', self.data_test[0].shape, self.data_test[1].shape)

        tr_batch_num = int(self.data_train[1].shape[0] / tr_batch_size)
        echo(3, tr_batch_num) 
        display_batch = int(tr_batch_num / display_num)  

        saver = tf.train.Saver(max_to_keep=10)  

        log(f'------- {time_str()} -------')

        for epoch in range(max_max_epoch):
            print(f'------  \033[92m{epoch} epochs \033[0m -------') 
            _lr = 0.01 if epoch < max_epoch else 0.005
            start_time = time.time()
            _losstotal, show_loss, best_dev_acc = 0.0, 0.0, -1

            for batch in range(tr_batch_num):  
                fetches = [self.model.loss, self.model.train_op]
                begin_index = batch * tr_batch_size
                end_index = (batch + 1) * tr_batch_size
                X_batch = self.data_train[0][begin_index:end_index]
                Y_batch = self.data_train[1][begin_index:end_index]
                echo(0, X_batch[57,0], Y_batch[57,0])

                feed_dict = {self.model.X_inputs: X_batch,
                            self.model.y_inputs: Y_batch,
                            self.model.lr: _lr,
                            self.model.batch_size: tr_batch_size,
                            self.model.keep_prob: 0.5}
                _loss, _ = sess.run(fetches, feed_dict)  
                _losstotal += _loss
                show_loss += _loss
                if not (batch + 1) % display_batch and not epoch:
                    train_p, train_r, train_macro_f1, _ = self.test_epoch(self.data_train, sess, 'Train')
                    dev_p, dev_r, dev_macro_f1, _ = self.test_epoch(self.data_dev, sess, 'Dev')
                    if dev_macro_f1 > best_dev_acc:
                        test_p, test_r, test_macro_f1, predict = self.test_epoch(self.data_test, sess, 'Test')
                        best_dev_acc = dev_macro_f1
                        log(f'{epoch}-{batch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}|{test_p:.2f}|{test_r:.2f}|{test_macro_f1:.2f}|')
                    else:
                        log(f'{epoch}-{batch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}|')

                    echo(f'training loss={show_loss / display_batch}')
                    show_loss = 0.0
            mean_loss = _losstotal / tr_batch_num
            if not (epoch + 1) % 1:
                save_path = saver.save(sess, self.model.model_save_path, global_step=(epoch + 1))
                print('the save path is ', save_path)
            if epoch % 1:
                train_p, train_r, train_macro_f1, _ = self.test_epoch(self.data_train, sess, 'Train')
                dev_p, dev_r, dev_macro_f1, _ = self.test_epoch(self.data_dev, sess, 'Dev') 
                log(f'{epoch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}|')
                if dev_macro_f1 > best_dev_acc:
                    test_p, test_r, test_macro_f1, predict = self.test_epoch(self.data_test, sess, 'Test')
                    best_dev_acc = dev_macro_f1

            echo(1, f'Training {self.data_train[1].shape[0]}, loss={mean_loss:g} ')
            echo(2, f'Epoch training {self.data_train[1].shape[0]}, loss={mean_loss:g}, speed={time.time() - start_time:g} s/epoch')

        log(f"Best Dev Macro_f1: {best_dev_acc:.2f}%")
        log(f"Best Test P: {test_p:.2f}%, R: {test_r:.2f}%, Macro_f1: {test_macro_f1:.2f}%")

        sess.close()
        return predict

    def test_epoch(self, dataset, sess, types:str):
        
        _batch_size = 1
        _y = dataset[1]
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)
        predict = []
        fetches = [self.model.scores, self.model.length, self.model.transition_params]

        for i in range(batch_num):
            begin_index = i * batch_num
            end_index = (i + 1) * batch_num
            X_batch = dataset[0][begin_index:end_index]
            Y_batch = dataset[1][begin_index:end_index]
            feed_dict = {self.model.X_inputs: X_batch,
                         self.model.y_inputs: Y_batch,
                         self.model.lr: 1e-5,
                         self.model.batch_size: _batch_size,
                         self.model.keep_prob: 1.0}

            test_score, test_length, transition_params = sess.run(fetches=fetches, feed_dict=feed_dict)
            # echo(0, np.array(test_score).shape)

            for tf_unary_scores_, y_, sequence_length_ in zip(test_score, Y_batch, test_length):
                tf_unary_scores_ = tf_unary_scores_
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transition_params)
                # print(np.array(viterbi_sequence).shape)
                predict.append(viterbi_sequence)
        if types == 'Test':
            pickle.dump(predict, open(f"{con.RESULT['CWS']}.pkl", 'wb'))
                  
        p, r, macro_f1 = evaluation(_y, predict, dataset[2], types)
        return p, r, macro_f1, predict

def test_params(num_seq: int, num_word: int, word_size: int, num_tag: int):
    x = np.random.randint(word_size, size=[num_seq, num_word]).astype(np.int32)
    y = np.random.randint(num_tag, size=[num_seq, num_word]).astype(np.int32)
    seq = np.random.randint(0, 5, size=[num_seq]).astype(np.int32)
    return x, y, seq


if __name__ == "__main__":
    num_seq = 10
    num_word = 20
    word_size = 3333
    num_tag = 4
    data_train = test_params(num_seq, num_word, word_size, num_tag)
    data_dev = test_params(10, num_word, word_size, num_tag)
    data_test = test_params(10, num_word, word_size, num_tag)

    model = BiLSTM_CRF_Model(max_len=num_word, 
                            vocab_size=word_size, 
                            num_tag=num_tag, 
                            model_save_path='./checkpoint/checkpoint', 
                            embed_size=256,  
                            hs=512)
    train = BiLSTMTrain(data_train, data_dev, data_test, model)
    train.train(100, 200, 5, 1)
    

