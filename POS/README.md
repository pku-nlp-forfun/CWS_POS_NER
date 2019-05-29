# Part-of-speech Tagging

## Approach

### Pure Mapping

Using the most frequent/common POS tagging table as the result.

## Reminder

* For the space character `$$_` or something that don't have POS tag but it may should have -_-". I'll set empty string `''` as its POS.

## Problem

### The NOT IN TRAINING SET Word

The following shit in the `testset1/test_pos1.txt` don't exist in `trainset/train_pos.txt`

```txt
10<sup>9<, 气管镜, 沙门菌株, 病情严重, 情况不明, 噻肟等, UA, 上式, 移项, UA, <sup>+, <sup>-, 五日, 风, 尖型, CCMD, Syndrome, 发病史, gllop, rhythm, P<sub>2<, 叉, 夹角, 夹角, 落入, 气陷, 补偿, 气道插管, 保暖箱, 发出报警, 盖, 数层, 布, 褥垫, 小流量, 血淋巴细胞, 丽, 华松, 惠珍, 非典型肺炎, 附6, 188, 188, pneumococcal, 铁锈色, 球状带, 状带, 中间层, 酮, 诸类, 羟, cytochrome, 超家族, 低密度, 条件者, 脱敏疗法, FMR, 发生变化, 传至, 时则, 传至, 时则, 缩减, 所携有, FMR, X染色体, 突变型, FMR, 突变型, FMR, 变化趋势, 缩减, 尚受, FMR, FMR, 定型化, 抽象思维, 性别差异, 流向, 流向, 静力, 流向, 剥, 叶间, 误伤, 粟粒, 积德, 122, 122, 保管, trachomatis, 滤泡性, 睑内翻, 睫毛, 经常性, 测者, 测处, 直角, 化学品, 可意料者, 消炎, 止痛药, 鸡胚羊膜腔, 阴转率, 泻药, 吐蛔虫, 碘酪氨酸, 侵扰, 侵害, 蛲虫病, enterobisis, 树芽, 珠样, 固有层, 相伴, 痒, 秃发, 脱发, 宽窄, 矮胖型, 封底, 彩图, 向内, 瘦长形, 封底, 彩图, 豚鼠, 荧光法, 阿拉斯加, 调查报告, 62, Ziegler, 84, 爱阿华州, 结果显示, 补充剂, 抗HBsAg, 酶标, 缩微版, 开具, 开具, 世界大战, 士兵们, 那时, 神经症, 那时, 战争, 平时期, 应激物, traumatic, stress, reserve, ERV, 闭陷, 谷草, 谷丙转氨酶, 几型, 颈面型放线菌, 胸部型放线菌, 腹部型放线菌, 皮肤型放线菌, 脑型放线菌, 放线菌性, 足菌病, 查血沉, 外斐, 抗溶血素链球菌, 精氨酸琥珀酸, 分别为, 515, PaCO<sub>2<, 从事, 从中, 得益, obstructive, 其实不然, 打鼾, 打鼾, 过去史, 前角, 中继站, 初级, 葡萄簇型, 粗暴, 肤色, 抽泣, 30秒, 疫情报告, 牛乳钙, 后下部, 冠状窦口, 右移位, Yokoyama, 脑叶, 脑叶, 篇幅, 细述, 简述, 甘油盐水, 积分子, 铁丝, 圈样, 核碎片, 苏木素, 病变者, 五年, 检查台, 台边, 疑诊, ABO血型, ABO血型, AB型, 析出, 摄水量, 尿胱氨酸, 应妩, 规程, 南京, 东南大学, 725, 内化, 谈论, 自由, 公正, 价值观, 张嘴, 含氯, 熏蒸, 有效氯, m<sup>3<, 含氯, 开门, streptococcal, pharyngitis, 磷脂酰甘油, 吸出液, 微泡计数法, 清洗液, 气泡, 大气泡, 排泄率, 血CO<sub>2<, 指着, 数3, 4件, 从头, 报出, 数1, 后来, 相加, 相加, 库, 盛行, MUD, 移除, MUD, 移植者, 菲薄, 菲薄, 毛细毛细血管瘤, 手背, 鼻腔黏膜, 指压, 电凝法, immune, 异己, 一般而言, 变态反应性, vibration, duodenalulcer, gastriculcer, 胃穿孔, 间区, 同胃镜, 遭遇, 目睹, 杀人, 战斗, 肺总量, TLC, VC, conventional, mechanical, 兼用型, ，也, PCV, ，也, assist, 带动, 道正压, patient, triggered, PTV, assist, intermittent, synchronised, intermittent, intermittent, 示意, 同一种, 独特性, 有别于, 其他型, 链反应, 温抗, AIHA者, 植入体, 食道, Mahan, 外周动脉, 矢状窦, <sub>4, 转铁蛋白, 楊, 丙昂, 执民, CO<sub>2<, 2.24, 症候群, sulfasalazine, 神经体液, 耽搁, 胃隐窝, 肾漏型, 普遍认为, α1-AT, 蛋白溶解酶, α1-AT, 蛋白溶解酶, 肺组织蛋白, CD3, <sup>+<, 后仰, 肾窝, 肾下垂, 肾下垂, 从来, 大概, 广泛应用, 围生, 多囊性, 发育不良, 多系, 肾下, 跨于, 之上, 男女比例, 一家, 几代, 尚待, CCMD, 垂体激素, m<sup>2<, 24h, 流应, 反流液, 数厘米, 一段, 面呈, 针尖状, 点影, 来回, 艳萍, 娟, .1988, 324, 324, 三核苷酸, 52, 精通, 主治医师, 麻醉科, 监护室, 本科, 麻醉科, 普外科, 放射科, 轮转, 毕业, 专业化, 床位, 护士长, 护师, 护士长, 职责, 运转, 配有, 工程师, 维修, 保养, 姊妹, 姊妹, 等臂, i, 荧光法, 交界线, 荧光带, 小钡条胶囊, 通过时间, 胜利, 给氧法, 维生素B<sub>1<, 肝胆系统, 硫胺素, 核对, 奚, 容平, cagA, vacA, 提问, 问答, 好奇性, 谈论, 果, 外在, 动物园, 脑出, 动静脉畸形, 多棘, 耳源性, 穿通, 最适, 温抗体, 冷抗体, IgG<sub>1<, IgG<sub>3<, IgG<sub>2<, IgG<sub>4<, 最适, 最适, 亚临床型, 乙患者, 血浆中因子, 炎型, 不符, 布鲁菌病, 肺吸虫, 相蛋白, 乳酸盐, 因子a, 子区, 598, 602, 尽少, 当事, 知情, 鼠伤寒沙门菌肠炎, propranolol, 氧需量, 阿替洛尔, atenolol, 美托洛尔, metoprolol, 品, 排泄性, 纵轴, 相交, 总阴, undetermined, anion, UA, 无机硫, 无机磷, 间液, Ca<sup>2+<, 细胞内液, 细胞内液, K<sup>+<, Ca<sup>2+<, Mg<sup>2+<, Na<sup>+<, K<sup>+<, 胶质错构瘤, 星形细胞瘤, 非钙化, 半透明肿瘤, 桑葚样, 色素缺失斑, 小眼球, 突眼, 磷酸吡哆醛, B<sub>6<, B<sub>6<, B<sub>6<, 少女, 两端, missense, mutation, 红细胞增多症, UA, 阳, UA-UC, 阳, Na<sup>+<, Cl<sup>-<, 拿走, 钱物, 偷别, 外出, 行窃, 行窃, 行窃, 违法, 总容量, 香精, 薄荷, 氨水, 胡椒粉, pharmacokinetics, 药效学, pharmacodynamics, 小肠结肠炎, 马富西, 萎靡不振, 快感, 小儿胰腺炎, 1个, 消散性, 胆管炎, 诺如病毒, 血尿素, 血尿素, 柏油样, 停留, 停留, 清除术, 限局性, 测温, 肛表, 监护室, 综合性, NaHCO<sub>3<, 前进方向, 探及, 由上而下, <sup>2, 缓解者, 刮取物, 抽吸液, 查多克, Chaddock, 专心, 听讲, 做事, 虎头蛇尾, 拖拉, 动机, 反义核苷, 相容, 规范性, 大室, 单纯房, 红细胞增多症, 收缩因子, 亚临床型, 眼角, 入脑量, 首量, 注速, 紫癜肾炎, 第5对, 5<sub>P<, 第5号, 5<sub>P<, sub>-, 0.01%, Peterson, Stakey, ∶1, 庆云, 0.242%, 组化, isolated, 膨出壁, 低密度, 蛇头样影, 蛇头样, 磷脂类, 堵闭, 调节性, CD4<sup>+<, 单细胞, 滑膜成纤维, E<sub>2<, 瀑布, 状体炎, 请, 弛缓性, 软瘫, 时肺, 内动, CcO2, CaO2, CcO2, CvO2, 可吸, 简略, PaO<sub>2<, 双氢克尿塞, 肝功, 抚养者, 清晰度, 出生史, 家庭史, 伸面, 腰椎脊, 铁一, 科学技术, 横结肠, 横结肠, 截断征, 轮廓不清, 肝肿大, 酮, 心肌灌注扫描, 心肌灌注不良, 冠状血管, 视乳头水肿, 较长, 主征, 四类, bFGF, 幼儿型, 染色体显性, 泌尿外科, 脓肿液, 癌瘤
```

## Performance

### Using Gold CWS (from POS file) and Test the Pure Mapping Model

> Fixed the `<sub> <sup>` tag problem, but the weird thing is all three value are the same. But if the CWS task is completely the same, maybe this is possible.

| Precision | Recall | F1 Score |
| --------- | ------ | -------- |
| 95.39     | 95.39  | 95.39    |

## Resources

* [NLP-progress - Part-of-speech tagging](http://nlpprogress.com/english/part-of-speech_tagging.html)

### Article

CRF

* [NLP Guide: Identifying Part of Speech Tags using Conditional Random Fields](https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31)

HMM

* [An introduction to part-of-speech tagging and the Hidden Markov Model](https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/)

Bi-LSTM + CRF

* [Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

Brief Description

* [POS tags](https://www.sketchengine.eu/pos-tags/)
* [NLP 笔记 - Part of speech tags](http://www.shuang0420.com/2017/02/24/NLP%20%E7%AC%94%E8%AE%B0%20-%20Part%20of%20speech%20tags/)

### Example

CRF

* [yanshao9798/tagger](https://github.com/yanshao9798/tagger) - A Joint Chinese segmentation and POS tagger based on bidirectional GRU-CRF

HMM

* [pjhanwar/POS-Tagger](https://github.com/pjhanwar/POS-Tagger)
* [lkmcl37/Chinese-POS-Tagger](https://github.com/lkmcl37/Chinese-POS-Tagger) - Chinese Part-of-Speech Tagger based on HMM and Viterbi Algorithm (Java)

### Others

#### Application

English

* [Parts-of-speech.Info](https://parts-of-speech.info/)
* [Stanford Log-linear Part-Of-Speech Tagger](https://nlp.stanford.edu/software/tagger.html)
