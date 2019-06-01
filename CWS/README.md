# Chinese Word Segmentation

## Traditional Machine Learning

- B: begin of a token
- M: middle of a token
- E: end of a token
- S: single character as a token
- (O: Out of tags)

## Approach

1. Data preprocessing into the classification problem
2. Word representation
   1. One-hot
   2. TF-IDF
   3. ...
3. Training model for some epochs
4. Evaluate in Dev croups
5. Predict in best evaluation epochs and generate result

## Experiment

| Model      | embed    | Train p | Train r | Train f1 | Dev p | Dev r | Dev f1 | Test p | Test n | Test f1 | Best Epoch |
| ---------- | -------- | ------- | ------- | -------- | ----- | ----- | ------ | ------ | ------ | ------- | ---------- |
| CRF        | One-Hot  | 98.86   | 99.19   | 99.03    | 76.36 | 75.06 | 75.70  | 78.54  | 75.68  | 77.08   | 130        |
| CRF        | Tf_idf   | 33.87   | 29.69   | 31.64    | 31.14 | 37.42 | 33.99  | 31.69  | 37.42  | 34.32   | 90         |
| CRF        | FastText | 30.24   | 33.21   | 31.65    | 34.59 | 37.29 | 35.89  | 36.13  | 37.99  | 37.03   | 50         |
| BiLSTM-CRF | One-Hot  | 94.46   | 95.19   | 94.83    | 90.02 | 91.08 | 90.55  | 91.01  | 91.32  | 91.16   | 8-75       |

### One typical

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

### F1_score not equal

check the script

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

### Appendix

#### TensorFlow CRF API

- [tensorflow/contrib/crf](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)
- [documentation](https://www.tensorflow.org/api_docs/python/tf/contrib/crf)
  - [tf.contrib.crf.crf_log_likelihood](https://www.tensorflow.org/api_docs/python/tf/contrib/crf/crf_log_likelihood)
  - [tf.contrib.crf.crf_decode](https://www.tensorflow.org/api_docs/python/tf/contrib/crf/crf_decode)

### TensorFlow BiLSTM Related API

- [TensorFlow Examples - bidirectional_rnn.py](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py)
- [RIP Tutorial - Creating a bidirectional LSTM](https://riptutorial.com/tensorflow/example/17004/creating-a-bidirectional-lstm)
- [tf.nn.bidirectional_dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)
- [tf.keras.layers.Bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)

### data

```plain txt
0|23.40|24.62|
10|40.06|34.21|
20|56.65|34.55|
30|46.59|57.50|
40|50.16|54.88|
50|46.53|57.44|
60|50.47|55.57|
70|66.09|58.75|
80|81.50||62.65|
90|96.89|71.10|
100|98.54|75.22|
110|99.03|75.70|
120|99.40|75.56|
130|99.69|75.53|
140|99.87|75.61|
150|99.94|75.65|
160|99.98|75.56|
170|99.99|75.59|
180|99.99|75.59|
190|99.99|75.55|
200|99.99|75.56|
210|99.99|75.57|
220|99.99|75.57|
230|99.99|75.57|
240|99.99|75.53|
250|99.99|75.49|
260|99.99|75.53|
270|99.99|75.50|
280|99.99|75.48|
290|99.99|75.49|
```
