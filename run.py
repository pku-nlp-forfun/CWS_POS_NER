from dataset import processing_pos_data, get_raw_article_from_cws_data
from constant import POS_DATA, CWS_DATA


def main():
    print('Data preprocessing for CWS and POS tagging...')
    cws_train, pos_dict = processing_pos_data(POS_DATA['Train'])
    cws_test, _ = processing_pos_data(POS_DATA['Test'])

    # Train
    print('Training CWS model...')

    # Test
    print('Generating result...')
    raw_article = get_raw_article_from_cws_data(CWS_DATA['Test'])

    # ONLY FOR TEST MUST REMOVE
    with open(CWS_DATA['Test'], 'r') as cws_f:
        as_cws_predict = cws_f.read()
    # TODO: use cws file as result to test the result of pos and test the output generater

    # Evaluate
    print('Evaluating...')


if __name__ == "__main__":
    main()
