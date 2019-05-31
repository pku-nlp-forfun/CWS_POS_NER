import constant as con
import numpy as np
import pickle

from CWS.cws_crf import crf_tf
from collections import Counter
from enum import Enum
from numba import jit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from typing import List, Dict
from util import echo


class EMBED_TYPE(Enum):
    ONE_HOT = 0
    TF_IDF = 1
    FAST_TEXT = 2


embed_type = EMBED_TYPE.FAST_TEXT


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
        if embed_type == EMBED_TYPE.FAST_TEXT:
            pre_set = [*train_set, *test_set, *dev_set]
        else:
            pre_set = train_set
        word_list = sum([[jj[0] for jj in ii] for ii in pre_set], [])
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

    def prepare_data(self, now_set: List, origin_set: List) -> (List, List, List):
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
        elif embed_type == EMBED_TYPE.FAST_TEXT:
            x = self.char_embed(x)

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
        for ii in n_gram_dict[0].keys():
            echo(0, transformed[0][ii])
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

        echo(1, len(no_exist_word), 'no exist Over')
        return n_gram_dict

    def char_embed(self, word_set: List):
        ''' char embed '''
        if embed_type == EMBED_TYPE.FAST_TEXT:
            embed_path = 'embedding/gigaword_chn.all.a2b.uni.ite50.vec'
        embed = self.load_embedding(embed_path)
        echo(1, 'len of embed', len(embed))
        word_set = np.array(word_set)
        num_fea = len(list(embed.values())[0])
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)
        word_set = word_set.reshape(-1)
        result_set = np.array([embed[ii] if ii in embed else np.zeros(
            num_fea) for ii in word_set])
        return result_set.reshape([num_seq, num_word, num_fea])

    def load_embedding(self, data_path: str) -> Dict[str, List[float]]:
        ''' load embedding '''
        with open(data_path) as f:
            origin_embed = [ii.strip() for ii in f.readlines()]
        origin_embed = [ii for ii in origin_embed if ii.split(' ')[
            0] in self.word2id]
        embed = {self.word2id[ii.split(' ')[0]]: np.array(
            ii.split(' ')[1:]).astype(np.float16) for ii in origin_embed}
        return embed

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
    def __init__(self, pos_counter_dict: dict):
        self.knowledge_graph = pos_counter_dict

    def predict_pos(self, word: str) -> str:
        if word in ('', '\n'):
            return ''
        try:
            return self.knowledge_graph[word].most_common()[0][0]
        except:  # every word not in the training set we give it Noun = =
            return 'n'

    def predict_list(self, words: list) -> list:
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
