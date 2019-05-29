from dataset import processing_pos_data, get_raw_article_from_cws_data
from constant import POS_DATA, CWS_DATA, RESULT
from evaluate import total_evaluate
from model import CWSModel, POSModel
from POS.pos_data import load_cws_result_as_input, from_pos_list_to_file, load_cws_result_from_pos_as_input
import os


def pos_test_by_cws_gold(pos_dict):
    # ONLY FOR TEST MUST REMOVE
    # This shit is insane = =
    # 10<sup>9<> WTF????????? but in POS_DATA['POS'] is correct....zzzz
    # cws_gold = load_cws_result_as_input(CWS_DATA['Test'])
    cws_gold = load_cws_result_from_pos_as_input(POS_DATA['Test'])
    pos_model = POSModel(pos_dict)
    pos_predict = pos_model.predict_all(cws_gold)
    from_pos_list_to_file(pos_predict, RESULT['POS'])


def main():
    print('Data preprocessing for CWS and POS tagging...')
    cws_train, pos_dict = processing_pos_data(POS_DATA['Train'])
    cws_dev, _ = processing_pos_data(POS_DATA['Dev'])
    cws_test, _ = processing_pos_data(POS_DATA['Test'])
    print('Data over...')
    cws = CWSModel(cws_train, cws_dev, cws_test)
    print('Over')
    cws.run_model()

    # Train
    print('Training CWS model...')

    # Test
    print('Generating result...')
    raw_article = get_raw_article_from_cws_data(CWS_DATA['Test'])

    # ONLY FOR TEST MUST REMOVE
    pos_test_by_cws_gold(pos_dict)

    # Evaluate
    print('Evaluating...')
    total_evaluate(CWS_DATA['Test'], CWS_DATA['Test'],
                   POS_DATA['Test'], RESULT['POS'], verbose=True)


if __name__ == "__main__":
    os.makedirs('Result', exist_ok=True)
    main()
