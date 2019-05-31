import numpy as np

from CWS.cws_crf import fastF1
from Evaluation.pos_evaluate import score
from constant import CWS_DATA, POS_DATA, CWS_LAB2ID, RESULT
from dataset import processing_pos_data
from util import log, echo


def cws_scorer(predict: str, goal: str, verbose: bool = True):
    refs = [ii.strip() for ii in open(goal, encoding='utf-8').readlines()]
    cads = [ii.strip() for ii in open(predict, encoding='utf-8').readlines()]
    # this verbose means return more result
    word_precision, word_recall, word_fmeasure, _, _, _, word_tnr, _ = score(refs, cads, verbose=True, is_cws=True)
      
    if verbose:
        log(f'|{word_precision * 100:.2f}|{word_recall * 100:.2f}|{word_fmeasure * 100:.2f}|')
        print(f'Word precision: {word_precision * 100:.2f}%')
        print(f'Word recall: {word_recall * 100:.2f}%')
        print(f'Word F-measure: {word_fmeasure * 100:.2f}%')

    return word_precision, word_recall, word_fmeasure, word_tnr


def pos_scorer(predict: str, goal: str, verbose: bool = True):
    refs = open(goal,
                encoding='utf-8').readlines()
    cads = open(predict, encoding='utf-8').readlines()
    _, _, _, tag_precision, tag_recall, tag_fmeasure, _, tag_tnr = score(
        refs, cads, verbose=True)  # this verbose means return more result
    log(f'|{tag_precision * 100:.2f}|{tag_recall * 100:.2f}|{tag_fmeasure * 100:.2f}|')
    if verbose:
        print(f'Tag precision: {tag_precision * 100:.2f}%')
        print(f'Tag recall: {tag_recall * 100:.2f}%')
        print(f'Tag F-measure: {tag_fmeasure * 100:.2f}%')

    return tag_precision, tag_recall, tag_fmeasure, tag_tnr


def total_evaluate(cws_predict: str, cws_goal: str, pos_predict: str, pos_goal: str, output: str = 'Evaluation.md', verbose=False):
    word_precision, word_recall, word_fmeasure, _ = cws_scorer(
        cws_predict, cws_goal, verbose)
    tag_precision, tag_recall, tag_fmeasure, _ = pos_scorer(
        pos_predict, pos_goal, verbose)
    with open(output, 'w') as result:
        result.write('# Evaluation Result (in %)\n\n')
        result.write('## CWS\n\n')
        result.write('Precision|Recall|F1 Score\n')
        result.write('---------|------|--------\n')
        result.write('%9.2f|%6.2f|%8.2f\n\n' %
                     (word_precision*100, word_recall*100, word_fmeasure*100))

        result.write('## POS Tagging\n\n')
        result.write('Precision|Recall|F1 Score\n')
        result.write('---------|------|--------\n')
        result.write('%9.2f|%6.2f|%8.2f\n' %
                     (tag_precision*100, tag_recall*100, tag_fmeasure*100))


def test_evaluation():
    echo(3, '-----_- Begin to test evaluation script -------')
    origin_test_set, _ = processing_pos_data(POS_DATA['Test'])
    # origin_test_set = [origin_test_set[0]] # test for first row
    seq = [len(ii) for ii in origin_test_set]
    y = sum([[int(CWS_LAB2ID[jj[1]] > 1) for jj in ii] for ii in origin_test_set], [])
    y_pre = np.random.randint(4, size=[len(y)])
    y_pre = [int(ii > 1) for ii in y_pre]

    ''' evaluation for f1 score '''
    change_idx, idx = [], -1
    for ii in seq:
        change_idx.append(ii + idx)
        idx += ii
    for ii in change_idx:
        y_pre[ii] = 1
    p, r, macro_f1 = fastF1(y, y_pre)
    echo(1, f"P: {p:.2f}%, R: {r:.2f}%, Macro_f1: {macro_f1:.2f}%")
    
    idx, test_predict_text = 0, []
    for ii in origin_test_set:
        temp_len = len(ii)
        temp_tag = y_pre[idx: idx + temp_len]
        temp_text = ''.join([f'{kk[0]}{" " if temp_tag[jj] else ""}' for jj, kk in enumerate(ii)]).strip()
        test_predict_text.append(temp_text)
        idx += temp_len
    with open(RESULT['CWS'], 'w') as f:
        f.write('\n'.join(test_predict_text))
    total_evaluate(RESULT['CWS'], CWS_DATA['Test_POS'], POS_DATA['Train'], POS_DATA['Train'], verbose=True)
    


if __name__ == "__main__":
    cws_scorer(CWS_DATA['Train'], CWS_DATA['Train'])
    pos_scorer(POS_DATA['Train'], POS_DATA['Train'])
    total_evaluate(CWS_DATA['Train'], CWS_DATA['Train'],
                   POS_DATA['Train'], POS_DATA['Train'])
    test_evaluation()
