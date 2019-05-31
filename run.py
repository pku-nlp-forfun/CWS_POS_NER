from dataset import processing_pos_data, get_raw_article_from_cws_data, get_raw_article
from constant import POS_DATA, CWS_DATA, RESULT, Final
from evaluate import total_evaluate
from model import CWSModel, POSModel
from POS.pos_data import load_cws_result_as_input, from_pos_list_to_file, load_cws_result_from_pos_as_input
import os
from enum import Enum


class MODE:
    train_evaluate = 0
    test_pos_only = 1
    predict_final = 2


CURRENT_MODE = MODE.train_evaluate


def pos_test_by_cws_gold(pos_data_set: str, pos_model):
    # This shit is insane = =
    # 10<sup>9<> WTF????????? but in POS_DATA['POS'] is correct....zzzz
    # cws_gold = load_cws_result_as_input(CWS_DATA['Test'])
    pos_data = POS_DATA[pos_data_set]
    cws_gold = load_cws_result_from_pos_as_input(pos_data)
    pos_predict = pos_model.predict_all(cws_gold)
    from_pos_list_to_file(pos_predict, RESULT['POS' + pos_data_set])
    total_evaluate(CWS_DATA['Test'], CWS_DATA['Test'],  # just add a arbitrary cws data
                   RESULT['POS' + pos_data_set], pos_data, verbose=True)


def main(mode=MODE.train_evaluate):
    print('Data preprocessing for CWS and POS tagging...')
    cws_train, pos_dict = processing_pos_data(POS_DATA['Train'])
    cws_dev, _ = processing_pos_data(POS_DATA['Dev'])
    cws_test, _ = processing_pos_data(POS_DATA['Test'])
    print('Generate models...')
    if mode != MODE.test_pos_only:
        cws_model = CWSModel(cws_train, cws_dev, cws_test)
    pos_model = POSModel(pos_dict, use_rule=True)

    # Train
    if mode != MODE.test_pos_only:
        print('Training CWS model...')
        cws_model.run_model()

    if mode == MODE.train_evaluate:
        print('Generating result...')
        raw_article = get_raw_article_from_cws_data(CWS_DATA['Test'])
        # TODO
    elif mode == MODE.predict_final:
        # Submission
        print('Generating result...')
        raw_article = get_raw_article()
        # TODO

    if mode == MODE.test_pos_only:
        print('Testing POS result on Train set...')
        pos_test_by_cws_gold('Train', pos_model)
        print('Testing POS result on Dev set...')
        pos_test_by_cws_gold('Dev', pos_model)
        print('Testing POS result on Test set...')
        pos_test_by_cws_gold('Test', pos_model)
        exit(0)

    if mode == MODE.train_evaluate:
        # Evaluate
        print('Evaluating...')
        total_evaluate(RESULT['CWS'], CWS_DATA['Test_POS'],
                       RESULT['POS'], POS_DATA['Test'], verbose=True)


if __name__ == "__main__":
    os.makedirs('Result', exist_ok=True)
    main(CURRENT_MODE)
