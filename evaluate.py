from Evaluation.pos_evaluate import score
from constant import CWS_DATA, POS_DATA
from util import log


def cws_scorer(predict: str, goal: str, verbose: bool = True):
    refs = [ii.strip() for ii in open(goal, encoding='utf-8').readlines()]
    cads = [ii.strip() for ii in open(predict, encoding='utf-8').readlines()]
    # this verbose means return more result
    word_precision, word_recall, word_fmeasure, _, _, _, word_tnr, _ = score(refs, cads, verbose=True, is_cws=True)
      
    if verbose:
        log(f'|{word_precision * 100:.2f}|{word_recall * 100:.2f}|{word_fmeasure * 100:.2f}|')
        print('Word precision:', word_precision)
        print('Word recall:', word_recall)
        print('Word F-measure:', word_fmeasure)

    return word_precision, word_recall, word_fmeasure, word_tnr


def pos_scorer(predict: str, goal: str, verbose: bool = True):
    refs = open(goal,
                encoding='utf-8').readlines()
    cads = open(predict, encoding='utf-8').readlines()
    _, _, _, tag_precision, tag_recall, tag_fmeasure, _, tag_tnr = score(
        refs, cads, verbose=True)  # this verbose means return more result
    if verbose:
        print('Tag precision:', tag_precision)
        print('Tag recall:', tag_recall)
        print('Tag F-measure:',  tag_fmeasure)

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


if __name__ == "__main__":
    cws_scorer(CWS_DATA['Train'], CWS_DATA['Train'])
    pos_scorer(POS_DATA['Train'], POS_DATA['Train'])
    total_evaluate(CWS_DATA['Train'], CWS_DATA['Train'],
                   POS_DATA['Train'], POS_DATA['Train'])
