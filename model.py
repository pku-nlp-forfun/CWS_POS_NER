import constant as con
import numpy as np
import pickle

from CWS.cws_crf import crf_tf
from enum import Enum
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from typing import List
from util import echo


class EMBED_TYPE(Enum):
    ONE_HOT = 0
    TF_IDF = 1


class CWSModel:
    ''' CRF for word segment '''

    def __init__(self, train_set: List, dev_set: List, test_set: List):
        self.MAX_LEN = 0
        self.origin_train_set = train_set
        self.origin_dev_set = dev_set
        self.origin_test_set = test_set
        self.statistical_data(train_set, dev_set, test_set)

    def statistical_data(self, train_set: List, dev_set: List, test_set: List, do_reshape: bool=True):
        ''' statistical data '''
        word_list = sum([[jj[0] for jj in ii] for ii in train_set], [])
        word_set = ['[OOV]', *list(set(word_list))]
        echo(1, len(word_list))
        word2id = {jj: ii for ii, jj in enumerate(word_set)}

        if not do_reshape:
            train_set = [[(word2id[jj] if jj in word2id else 0, con.CWS_LAB2ID[kk])
                        for jj, kk in ii] for ii in train_set]
            dev_set = [[(word2id[jj] if jj in word2id else 0, con.CWS_LAB2ID[kk])
                        for jj, kk in ii] for ii in dev_set]
            test_set = [[(word2id[jj] if jj in word2id else 0, con.CWS_LAB2ID[kk])
                         for jj, kk in ii] for ii in test_set]
            self.train_set = train_set
            self.dev_set = dev_set
            self.test_set = test_set
        else:
            ''' a way to reduce memory using '''
            self.word2id = word2id
            self.train_set = self.reshape_data(train_set)
            self.dev_set = self.reshape_data(dev_set)
            self.test_set = self.reshape_data(test_set)


    def reshape_data(self, origin_set: List, MAX_LEN: int = 200) -> List:
        ''' reshape data '''
        data_set = sum([[(self.word2id[jj] if jj in self.word2id else 0, con.CWS_LAB2ID[kk])
                         for jj, kk in ii] for ii in origin_set], [])
        data_len = len(data_set)
        pad_len = MAX_LEN - data_len % MAX_LEN
        echo(2, data_len, pad_len)
        data_set = np.array([*data_set, *[(0, 0)] * pad_len])
        reshape_data = data_set.reshape([MAX_LEN, len(data_set) // MAX_LEN, 2])
        if not pad_len:
            last_id = reshape_data.shape[0] - 1
            reshape_data = [jj if ii != last_id else jj[:MAX_LEN - pad_len]
                            for ii, jj in enumerate(reshape_data)]
        return reshape_data

    def prepare_data(self, now_set: List, origin_set:List, embed_type: EMBED_TYPE = EMBED_TYPE.ONE_HOT) -> (List, List, List):
        ''' prepare_data '''
        MAX_LEN = max([len(ii) for ii in now_set])
        print(f'MAX_LEN{MAX_LEN}')
        seq_len = [len(ii) for ii in now_set]
        seq = [len(ii) for ii in origin_set]
        x = self.pad_pattern(now_set, 0, MAX_LEN)
        y = self.pad_pattern(now_set, 1, MAX_LEN)
        if embed_type == EMBED_TYPE.ONE_HOT:
            x = self.one_hot(x)
        return x, y, seq_len, seq

    def pad_pattern(self, origin_set: List, idx: int, MAX_LEN: int) -> List:
        ''' pad pattern '''
        return [[jj[idx] for jj in ii] + [0] * (MAX_LEN - len(ii)) for ii in origin_set]

    def one_hot(self, word_set: List):
        ''' one hot embed '''
        word_set = np.array(word_set)
        num_fea = len(self.word2id)
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)

        return np.squeeze(np.eye(num_fea)[word_set.reshape(-1)]).reshape([num_seq, num_word, num_fea])

    def tf_idf(self, word_set:List):
        ''' tf-idf embed'''
        word_set = np.array(word_set)
        num_fea = len(self.word2id)
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)
        to_pipeline = [DictVectorizer(), TfidfTransformer()]
        data_transformer = make_pipeline(*to_pipeline)
        transformed = data_transformer.fit_transform(word_set.reshape(-1)).todense()

    def run_model(self):
        ''' run crf model '''
        train_x, train_y, train_seq, train_se = self.prepare_data(self.train_set, self.origin_train_set)
        dev_x, dev_y, dev_seq, dev_se = self.prepare_data(self.dev_set, self.origin_dev_set)
        test_x, test_y, test_seq, test_se = self.prepare_data(self.test_set, self.origin_test_set)
        print('Prepare Over')
        test_predict = crf_tf(train_x, train_y, train_seq, train_se, dev_x, dev_y, dev_seq, dev_se, test_x, test_y, test_seq, test_se, len(con.CWS_LAB2ID))
        test_predict = test_predict.reshape(-1)
        idx, test_predict_text = 0, []
        for ii in self.origin_test_set:
            temp_len = len(ii)
            temp_tag = test_predict[idx: idx + temp_len]
            temp_text = ''.join([f'{kk[0]}{"" if temp_tag[jj] < 2 else " "}' for jj, kk in enumerate(ii)]).strip()
            test_predict_text.append(temp_text)
            idx += temp_len
        with open(con.RESULT['CWS'], 'w') as f:
            f.write('\n'.join(test_predict_text))


# Knowledge graph for POS tagging
class POSModel:
    def __init__(self, pos_counter_dict: dict):
        self.knowledge_graph = pos_counter_dict

    def predict_pos(self, word: str)->str:
        if word in ('', '\n'):
            return ''
        try:
            return self.knowledge_graph[word].most_common()[0][0]
        except:  # every word not in the training set we give it Noun = =
            return 'n'

    def predict_list(self, words: list)->list:
        result = []
        for word in words:
            pos = self.predict_pos(word)
            if pos:
                result.append(word + '/' + pos)
            else:
                result.append(word)
        return result

    def predict_all(self, predict_sentence_list: list):
        result = []
        for sentence_list in predict_sentence_list:
            result.append(self.predict_list(sentence_list))
        return result
