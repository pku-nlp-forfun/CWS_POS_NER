import constant as con
import numpy as np
import pickle
import re
from random import random

from CWS.cws_crf import crf_tf
from collections import Counter
from enum import Enum
from numba import jit
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

    def statistical_data(self, train_set: List, dev_set: List, test_set: List, do_reshape: bool = True):
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

    def prepare_data(self, now_set: List, origin_set: List, embed_type: EMBED_TYPE = EMBED_TYPE.TF_IDF) -> (List, List, List):
        ''' prepare_data '''
        MAX_LEN = max([len(ii) for ii in now_set])
        print(f'MAX_LEN: {MAX_LEN}')
        seq_len = [len(ii) for ii in now_set]
        seq = [len(ii) for ii in origin_set]
        x = self.pad_pattern(now_set, 0, MAX_LEN)
        y = self.pad_pattern(now_set, 1, MAX_LEN)
        if embed_type == EMBED_TYPE.ONE_HOT:
            x = self.one_hot(x)
        elif embed_type == EMBED_TYPE.TF_IDF:
            x = self.tf_idf(x, seq)
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

        return np.squeeze(np.eye(num_fea)[word_set.reshape(-1)]).reshape([num_seq, num_word, num_fea]).astype(np.int16)

    def tf_idf(self, word_set: List, seq: List, n_gram: int = 4):
        ''' tf-idf embed'''

        word_set = np.array(word_set)
        num_fea = len(self.word2id)
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)
        origin_set_one = word_set.reshape(-1)

        n_gram_dict = self.prepare_n_gram(origin_set_one, seq, n_gram)
        n_gram_dict += [{}] * (num_seq * num_word - len(n_gram_dict))

        to_pipeline = [DictVectorizer(), TfidfTransformer()]
        data_transformer = make_pipeline(*to_pipeline)
        transformed = np.array(data_transformer.fit_transform(
            n_gram_dict).todense(), dtype=np.float16)
        echo(1, 'Tf idf Over')
        return transformed.reshape([num_seq, num_word, num_fea])

    @jit
    def prepare_n_gram(self, origin_set_one: List, seq: List, n_gram: int = 4):
        ''' prepare n gram'''
        idx, origin_set, n_gram_dict, exist_word, no_exist_word = 0, [], [], {}, []
        for ii in seq:
            origin_set.append(list(origin_set_one[idx:idx+ii]))
            idx += ii
        echo(1, 'Seq Length Over')
        for ii in origin_set:
            for jj, _ in enumerate(ii):
                t_seq_len = len(ii)
                begin_idx = max(0, jj - n_gram)
                end_idx = min(t_seq_len, jj + n_gram)
                n_gram_word = ii[begin_idx:end_idx]
                n_gram_count = dict(Counter(n_gram_word))
                n_gram_dict.append(n_gram_count)
                for kk, mm in n_gram_count.items():
                    exist_word[kk] = mm
        echo(1, 'n_gram Over')
        for ii in self.word2id.values():
            if ii not in exist_word:
                no_exist_word.append(ii)
        for ii in no_exist_word:
            n_gram_dict[-1][ii] = 0

        echo(1, 'no exist Over')
        return n_gram_dict

    def run_model(self):
        ''' run crf model '''
        train_x, train_y, train_seq, train_se = self.prepare_data(
            self.train_set, self.origin_train_set)
        dev_x, dev_y, dev_seq, dev_se = self.prepare_data(
            self.dev_set, self.origin_dev_set)
        test_x, test_y, test_seq, test_se = self.prepare_data(
            self.test_set, self.origin_test_set)
        print('Prepare Over')
        test_predict = crf_tf(train_x, train_y, train_seq, train_se, dev_x, dev_y,
                              dev_seq, dev_se, test_x, test_y, test_seq, test_se, len(con.CWS_LAB2ID))
        test_predict = test_predict.reshape(-1)
        idx, test_predict_text = 0, []
        for ii in self.origin_test_set:
            temp_len = len(ii)
            temp_tag = test_predict[idx: idx + temp_len]
            temp_text = ''.join(
                [f'{kk[0]}{"" if temp_tag[jj] < 2 else " "}' for jj, kk in enumerate(ii)]).strip()
            test_predict_text.append(temp_text)
            idx += temp_len
        with open(con.RESULT['CWS'], 'w') as f:
            f.write('\n'.join(test_predict_text))


# Knowledge graph for POS tagging
class POSModel:
    def __init__(self, pos_counter_dict: dict, use_rule: bool = True, test_mode: bool = False):
        self.knowledge_graph = pos_counter_dict
        self.__use_rule = use_rule
        self.__test_mode = test_mode

        self.__re_chinese = re.compile("([\u4E00-\u9FD5]+)")
        self.__re_entire_eng = re.compile('^[a-zA-Z]+$', re.U)
        self.__re_digit = re.compile("[\.0-9]+%?")  # 87 or 87% or 87.87%

    def __exception_handler(self, not_in_train_word: str):
        flag = 'n'

        if not self.__re_chinese.match(not_in_train_word):  # not chinese
            if self.__re_digit.match(not_in_train_word):  # digit
                flag = 'm'
            elif self.__re_entire_eng.match(not_in_train_word):  # english
                if not_in_train_word in ('kg', 'kPa', 'd', 'm', 'mg', 'mm', 'L', 'mmol', 'mol', 'mmHg', 'ml', 'pH', 'ppm', 'cm', 'cmH', 'g', 'km', 'sec'):
                    flag = 'q'  # a unit
                elif not_in_train_word[0].isupper() and not_in_train_word[1:].islower():
                    flag = 'nr' if random() > 0.5 else 'ns'  # must be a name or a place = =
                else:
                    flag = 'nx'
        # end with 状 but not 症状
        elif len(not_in_train_word) >= 2 and not_in_train_word[-1] == '状' and not_in_train_word[-2:] != '症状':
            flag = 'b'
        # end with 性, and not 毒性
        elif len(not_in_train_word) >= 2 and not_in_train_word[-1] == '性' and (not_in_train_word[-2:] != '毒性' and len(not_in_train_word) == 2):
            flag = 'b'

        return flag

    def predict_pos(self, word: str)->str:
        if word in ('', '\n'):
            return ''
        try:
            return self.knowledge_graph[word].most_common()[0][0]
        except:  # every word not in the training set we give it Noun = =
            flag = 'n'
            if self.__use_rule:
                flag = self.__exception_handler(word)
            if self.__test_mode:
                print((word, flag), end=', ')
            return flag

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


if __name__ == "__main__":
    from dataset import processing_pos_data
    from constant import POS_DATA
    from POS.pos_data import load_cws_result_from_pos_as_input, from_pos_list_to_file
    _, pos_dict = processing_pos_data(POS_DATA['Train'])
    cws_dev = load_cws_result_from_pos_as_input(POS_DATA['Dev'])
    cws_test = load_cws_result_from_pos_as_input(POS_DATA['Test'])
    pos_model = POSModel(pos_dict, test_mode=True)
    cws_not_train = [*cws_dev, *cws_test]
    pos_model.predict_all(cws_not_train)
