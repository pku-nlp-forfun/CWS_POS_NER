# Transfer dataset into traninable format
from constant import POS_DATA, CWS_DATA
from collections import defaultdict, Counter


def processing_pos_data(path: str = POS_DATA['Train']):

    dataset_list = []
    # Entry: List of a sentence
    # In a sentence, there are multiple tuple
    # [('我', S),
    #  ('是', S),
    #  ('大', B),
    #  ('非', M),
    #  ('洲', E)]
    pos_knowledge_graph = defaultdict(Counter)

    with open(path, 'r') as data_file:
        raw_data_lines = data_file.readlines()

    for raw_sentence in raw_data_lines:
        sentence_list = []
        for raw_word_tag in raw_sentence.split():
            this_is_shit = False
            try:
                word, pos = raw_word_tag.rsplit('/', 1)
            except:  # idiot dataset = =
                # $$_ and the innocent English word being seperated
                word = raw_word_tag
                pos = ''
                # this_is_shit = True

            # Into trainable format for CWS #

            if len(word) == 1:  # single word
                label = 'S'
                sentence_list.append((word, label))
            elif this_is_shit:  # currently don't use
                # ('$$_', 'S')
                label = 'S'  # or maybe O?!
                sentence_list.append((word, label))
            else:
                ('$', 'B'), ('$', 'M'), ('_', 'E')
                for i, char in enumerate(word):
                    if i == 0:
                        label = 'B'
                    elif i == len(word)-1:
                        label = 'E'
                    else:
                        label = 'M'
                    sentence_list.append((char, label))

            # POS knowledge graph #
            pos_knowledge_graph[word].update([pos])

        dataset_list.append(sentence_list)

    return dataset_list, pos_knowledge_graph


def get_raw_article_from_cws_data(path: str = CWS_DATA['Train'], output_path: str = ''):
    with open(path, 'r') as f_in:
        raw_data_with_space = f_in.read()

    # remove all the space
    raw_article = "".join(raw_data_with_space.split(' '))

    if output_path:
        with open(output_path, 'w') as f_out:
            f_out.write(raw_article)

    return raw_article


if __name__ == "__main__":
    # just for test purpose
    dataset_list, pos_knowledge_graph = processing_pos_data(POS_DATA['Train'])
    print(dataset_list[0])
    for w in pos_knowledge_graph.keys():
        print(w, pos_knowledge_graph[w].most_common()[0][0])
        break
    # Total words without duplicate in training set: 21247
    print(len(pos_knowledge_graph))
    get_raw_article_from_cws_data(CWS_DATA['Train'], 'Data/raw_article.txt')
