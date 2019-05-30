# Chinese Word Segmentation

## Traditional Machine Learning

- B: begin of a token
- M: middle of a token
- E: end of a token
- S: single character as a token
- (O: Out of tags)

## Bi-LSTM with CRF

![Bi-LSTM with CRF](https://pic1.zhimg.com/80/v2-aad7ef8156b33c51efeb0f7f4b6f614d_hd.jpg)

## Approach

1. Data preprocessing into the classification problem
2. Word representation
   1. One-hot
   2. TF-IDF
   3. ...
3. Training model for some epochs
4. Evaluate in Dev croups
5. Predict in best evaluation epochs and generate result

## Trouble

### Memory error

There is the significant trouble in our work.

In the design processing, we make the sequence pad to `MAX_LEN` sequence.

But in the actual, we find that it will be memory error if we pad to max sequence.

There are 513878 char in our Train Set.

But if we pad the seq to the MAX_LEN, it will change to 6106 \* 1060 = 6472360.

And if we use one-hot to do the word representation, it will cost 6106 seq \* 1060 char \* 2849 dim \* int16 = 68GB

In fact, the memory will bigger than 120GB.

So the only way is to reshape the sequence.

### tf.constant not support to bigger than 2GB

[Initializing tensorflow Variable with an array larger than 2GB](https://stackoverflow.com/questions/35394103/initializing-tensorflow-variable-with-an-array-larger-than-2gb)

```python
train_x_init = tf.placeholder(tf.float32, shape=np.array(train_x).shape)
train_x_t = tf.Variable(train_x_init)
session.run(tf.global_variables_initializer(), feed_dict={train_x_init: train_x})
```

## Resources

### Corpus

- [icwb2-data - Second International Chinese Word Segmentation Bakeoff 2 Data](http://sighan.cs.uchicago.edu/bakeoff2005/)
- [ZHWiki latest pages articles](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)

### Examples

Pure CRF

- [guokr/gkseg](https://github.com/guokr/gkseg) - Yet another Chinese word segmentation package based on character-based tagging heuristics and CRF algorithm

Bi-LSTM

- [**EricLingRui/NLP-tools**](https://github.com/EricLingRui/NLP-tools)
- [chantera/blstm-cws](https://github.com/chantera/blstm-cws)
  - [data/download.sh](https://github.com/chantera/blstm-cws/blob/master/data/download.sh)
- [luozhouyang/deepseg](https://github.com/luozhouyang/deepseg)

### Appendix - Tensorflow CRF API

- [tensorflow/contrib/crf](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)
- [documentation](https://www.tensorflow.org/api_docs/python/tf/contrib/crf)
  - [tf.contrib.crf.crf_log_likelihood](https://www.tensorflow.org/api_docs/python/tf/contrib/crf/crf_log_likelihood)
  - [tf.contrib.crf.crf_decode](https://www.tensorflow.org/api_docs/python/tf/contrib/crf/crf_decode)
