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

## Experiment


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

```
[[('2', '-NONE-'),
  ('.', '-NONE-'),
  ('近', '-NONE-'),
  ('2', '-NONE-'),
  ('周', '-NONE-'),
  ('内', '-NONE-'),
  ('有', '-NONE-'),
  ('过', '-NONE-'),
  ('严重', '-NONE-'),
  ('感染', '-NONE-'),
  ('，', '-NONE-'),
  ('或', '-NONE-'),
  ('WBC', '-NONE-'),
  ('＜', '-NONE-'),
  ('4', '-NONE-'),
  ('×', '-NONE-'),
  ('10<sup>9<', 'sup>'),
  ('/', '/'),
  ('L', '-NONE-'),
  ('，', '-NONE-'),
  ('或', '-NONE-'),
  ('对', '-NONE-'),
  ('CTX', '-NONE-'),
  ('过敏', '-NONE-'),
  ('，', '-NONE-'),
  ('或', '-NONE-'),
  ('2', '-NONE-'),
  ('周', '-NONE-'),
  ('内', '-NONE-'),
  ('用', '-NONE-'),
  ('过', '-NONE-'),
  ('其他', '-NONE-'),
  ('细胞', '-NONE-'),
  ('免疫', '-NONE-'),
  ('抑制剂', '-NONE-'),
  ('，', '-NONE-'),
  ('重症', '-NONE-'),
  ('肾病', '-NONE-'),
  ('综合征', '-NONE-'),
  ('表现', '-NONE-'),
  ('，', '-NONE-'),
  ('血清白蛋白', '-NONE-'),
  ('＜', '-NONE-'),
  ('2', '-NONE-'),
  ('g', '-NONE-'),
  ('/', '/'),
  ('L', '-NONE-'),
  ('时', '-NONE-'),
  ('，', '-NONE-'),
  ('应', '-NONE-'),
  ('慎用', '-NONE-'),
  ('CTX', '-NONE-'),
  ('。', '-NONE-'),
  ('由于', '-NONE-'),
  ('儿童', '-NONE-'),
  ('SLE', '-NONE-'),
  ('的', '-NONE-'),
  ('发病', '-NONE-'),
  ('高峰', '-NONE-'),
  ('在', '-NONE-'),
  ('11', '-NONE-'),
  ('～', '-NONE-'),
  ('15', '-NONE-'),
  ('岁', '-NONE-'),
  ('，', '-NONE-'),
  ('因此', '-NONE-'),
  ('，', '-NONE-'),
  ('治疗', '-NONE-'),
  ('时', '-NONE-'),
  ('应该', '-NONE-'),
  ('考虑', '-NONE-'),
  ('青春期', '-NONE-'),
  ('发育', '-NONE-'),
  ('的', '-NONE-'),
  ('问题', '-NONE-'),
  ('。', '-NONE-'),
  ('目前', '-NONE-'),
  ('，', '-NONE-'),
  ('在', '-NONE-'),
  ('狼疮', '-NONE-'),
  ('肾炎', '-NONE-'),
  ('，', '-NONE-'),
  ('应用', '-NONE-'),
  ('CTX', '-NONE-'),
  ('冲击', '-NONE-'),
  ('治疗', '-NONE-'),
  ('尿蛋白', '-NONE-'),
  ('消失', '-NONE-'),
  ('后', '-NONE-'),
  ('可用', '-NONE-'),
  ('硫唑嘌呤', '-NONE-'),
  ('维持', '-NONE-'),
  ('，', '-NONE-'),
  ('剂量', '-NONE-'),
  ('为', '-NONE-'),
  ('每', '-NONE-'),
  ('日', '-NONE-'),
  ('1', '-NONE-'),
  ('～', '-NONE-'),
  ('2.5', '-NONE-'),
  ('mg', '-NONE-'),
  ('/', '/'),
  ('kg', '-NONE-'),
  ('。\n', '-NONE-')]]

  [[('2', '-NONE-'),
  ('.近', '-NONE-'),
  ('2', '-NONE-'),
  ('周', '-NONE-'),
  ('内', '-NONE-'),
  ('有', '-NONE-'),
  ('过', '-NONE-'),
  ('严重感', '-NONE-'),
  ('染', '-NONE-'),
  ('，或W', '-NONE-'),
  ('BC＜4×10', '-NONE-'),
  ('<s', '-NONE-'),
  ('up>', '-NONE-'),
  ('9<', ''),
  ('su', '-NONE-'),
  ('p>', '-NONE-'),
  ('/', '/'),
  ('L', '-NONE-'),
  ('，', '-NONE-'),
  ('或对CT', '-NONE-'),
  ('X', '-NONE-'),
  ('过', '-NONE-'),
  ('敏，或2周内', '-NONE-'),
  ('用', '-NONE-'),
  ('过', '-NONE-'),
  ('其他', '-NONE-'),
  ('细', '-NONE-'),
  ('胞免疫抑制剂，', '-NONE-'),
  ('重', '-NONE-'),
  ('症肾', '-NONE-'),
  ('病', '-NONE-'),
  ('综合征表', '-NONE-'),
  ('现，血', '-NONE-'),
  ('清', '-NONE-'),
  ('白', '-NONE-'),
  ('蛋白＜', '-NONE-'),
  ('2g', ''),
  ('L', '-NONE-'),
  ('时', '-NONE-'),
  ('，应慎', '-NONE-'),
  ('用C', '-NONE-'),
  ('T', '-NONE-'),
  ('X', '-NONE-'),
  ('。', '-NONE-'),
  ('由于儿童', '-NONE-'),
  ('S', '-NONE-'),
  ('LE', '-NONE-'),
  ('的发', '-NONE-'),
  ('病高', '-NONE-'),
  ('峰', '-NONE-'),
  ('在11～1', '-NONE-'),
  ('5岁', '-NONE-'),
  ('，', '-NONE-'),
  ('因', '-NONE-'),
  ('此，', '-NONE-'),
  ('治疗', '-NONE-'),
  ('时', '-NONE-'),
  ('应', '-NONE-'),
  ('该', '-NONE-'),
  ('考', '-NONE-'),
  ('虑青', '-NONE-'),
  ('春期', '-NONE-'),
  ('发育', '-NONE-'),
  ('的', '-NONE-'),
  ('问题。目', '-NONE-'),
  ('前，在', '-NONE-'),
  ('狼', '-NONE-'),
  ('疮', '-NONE-'),
  ('肾', '-NONE-'),
  ('炎，应', '-NONE-'),
  ('用', '-NONE-'),
  ('CT', '-NONE-'),
  ('X', '-NONE-'),
  ('冲击', '-NONE-'),
  ('治', '-NONE-'),
  ('疗尿', '-NONE-'),
  ('蛋', '-NONE-'),
  ('白', '-NONE-'),
  ('消', '-NONE-'),
  ('失', '-NONE-'),
  ('后可用', '-NONE-'),
  ('硫唑', '-NONE-'),
  ('嘌', '-NONE-'),
  ('呤', '-NONE-'),
  ('维', '-NONE-'),
  ('持，', '-NONE-'),
  ('剂', '-NONE-'),
  ('量', '-NONE-'),
  ('为', '-NONE-'),
  ('每', '-NONE-'),
  ('日', '-NONE-'),
  ('1', '-NONE-'),
  ('～2', '-NONE-'),
  ('.', '-NONE-'),
  ('5', '-NONE-'),
  ('m', '-NONE-'),
  ('g', 'k'),
  ('g。', '-NONE-')]]
```