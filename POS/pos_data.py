def load_cws_result_as_input(path: str):
    with open(path, 'r') as cws_file:
        cws_result_raw = cws_file.read()

    all_cws_word = [line.split(' ') for line in cws_result_raw.split('\n')]

    return all_cws_word


def load_cws_result_from_pos_as_input(path: str):
    with open(path, 'r') as pos_file:
        pos_result_raw = pos_file.read()

    all_cws_word = [[word_pos.rsplit(
        '/', 1)[0] for word_pos in line.split(' ')] for line in pos_result_raw.split('\n')]

    return all_cws_word


def from_pos_list_to_file(pos_sentence_list: list, output_path: str):
    with open(output_path, 'w') as output_file:
        for j, pos_sentence in enumerate(pos_sentence_list):
            for i, word_pos in enumerate(pos_sentence):
                output_file.write(word_pos)
                if i != len(pos_sentence) - 1:
                    output_file.write(' ')
            if j != len(pos_sentence_list) - 1:
                output_file.write('\n')
