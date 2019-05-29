# CWS/POS/NER

Chinese word segmentation, Part-of-speech tagging and Medical named entity recognition From scratch.

## Getting Started

Dependencies:

* tensorflow

```sh
# training, testing and evaluation
python3 run.py
```

### Structure

```txt
├── Data         => data set given by TA
│   ├── devset
│   ├── testset1
│   └── trainset
├── Evaluation   => eval scripts given by TA
|
├── CWS          => CWS model
├── POS          => POS tagging model
├── NER          => NER model
|
├── constant.py  => some global constants and variables
|
├── dataset.py   => data preprocessing
├── model.py     => high-level model API for all our model
├── evaluate.py  => high-level evaluation API
└── run.py       => the entire process
```

## Task Description

### Directory Structure

* Data: (each has its _cws, _pos, _ner file)
  * devset
  * testset1
  * trainset
* Evaluation
  * pos_evaluate.py
  * ner_evaluate.py

## Resources

### Article

* [**Chinese Word Segmentation: Another Decade Review (2007-2017)**](https://arxiv.org/pdf/1901.06079.pdf)
* [NLP 中文分詞 – 結巴](https://allenlu2007.wordpress.com/2018/06/16/nlp-%E4%B8%AD%E6%96%87%E8%A9%9E%E5%B5%8C%E5%85%A5-%E6%96%B7%E8%A9%9E%EF%BC%8F%E5%88%86%E8%A9%9E/)

### Paper

CWS

* [State-of-the-art Chinese Word Segmentation with Bi-LSTMs](https://arxiv.org/abs/1808.06511) - Google AI Language
* [Chinese Word Segmentation with Conditional Random Fields and Integrated Domain Knowledge](https://people.cs.umass.edu/~mccallum/papers/chineseseg.pdf)
* [A Conditional Random Field Word Segmenter for Sighan Bakeoff 2005](https://nlp.stanford.edu/pubs/sighan2005.pdf)

#### jieba

#### pkuseg

* [ACM Digital Library](https://dl.acm.org/citation.cfm?id=2390560)

```bibtex
@inproceedings{DBLP:conf/acl/SunWL12,
author = {Xu Sun and Houfeng Wang and Wenjie Li},
title = {Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection},
booktitle = {The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea- Volume 1: Long Papers},
pages = {253--262},
year = {2012}}
```

### Related Tools

#### CRF

* [tensorflow/contrib/crf](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - A fast implementation of Conditional Random Fields (CRFs)
  * [chokkan/crfsuite](https://github.com/chokkan/crfsuite)
* [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/)
  * [TeamHG-Memex/sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
