# Chinese Word Segmentation

## Traditional Machine Learning

* B: begin of a token
* M: middle of a token
* E: end of a token
* S: single character as a token
* (O: Out of tags)

## Bi-LSTM with CRF

![Bi-LSTM with CRF](https://pic1.zhimg.com/80/v2-aad7ef8156b33c51efeb0f7f4b6f614d_hd.jpg)

## Approach

1. Data preprocessing into the classification problem
2. Word representation
   1. One-hot
   2. TF-IDF
   3. ...
3. Training model
4. Predict and generate result

## Resources

### Corpus

* [icwb2-data - Second International Chinese Word Segmentation Bakeoff 2 Data](http://sighan.cs.uchicago.edu/bakeoff2005/)
* [ZHWiki latest pages articles](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)

### Examples

Pure CRF

* [guokr/gkseg](https://github.com/guokr/gkseg) - Yet another Chinese word segmentation package based on character-based tagging heuristics and CRF algorithm

Bi-LSTM

* [chantera/blstm-cws](https://github.com/chantera/blstm-cws)
  * [data/download.sh](https://github.com/chantera/blstm-cws/blob/master/data/download.sh)
* [luozhouyang/deepseg](https://github.com/luozhouyang/deepseg)
