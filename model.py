# CRF for word segment
class CWSModel:
    def __init__(self):
        pass


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
