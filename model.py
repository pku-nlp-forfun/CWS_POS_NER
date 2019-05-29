# CRF for word segment
class CWSModel:
    def __init__(self):
        pass


# Knowledge graph for POS tagging
class POSModel:
    def __init__(self, pos_counter_dict):
        self.knowledge_graph = pos_counter_dict

    def predict(self, word):
        return self.knowledge_graph[word].most_common()[0][0]
