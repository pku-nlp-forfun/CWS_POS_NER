# Part-of-speech Tagging

## Approach

### Pure Mapping

Using the most frequent/common POS tagging table as the result.

## Reminder

* For the space character `$$_` or something that don't have POS tag but it may should have -_-". I'll set empty string `''` as its POS.

## Problem

### The NOT IN TRAINING SET Word

The following shit in the `testset1/test_pos1.txt` and `devset/val_pos.txt` don't exist in `trainset/train_pos.txt`

```txt
Mayo, Clinic, 1950, 年间, Minnesota, 州, Olmsted, 郡, 0.26, 但至, 0.65%, 比为, 头胎, 最易, 黑种人, 幼时, 上腹, 可吐, 剑, 偏右, 局性, 窦血栓, 颈静脉血栓, 胜任, 主课, 课间, 美味, 维生素B<sub>2</sub>, 鲜奶, 绿叶, 黄红色, 细嚼慢咽, 饥饱, jaundice, 未结合胆红素, 极低出生体重儿, 截然, developmental, hyperbilirubinemia, 功效, 全杀菌药, 自始至终, exanthem, subitum, 玫瑰, roseola, infantum, 蛔虫性, MODY, 等渗盐水, 高渗糖水, 硫酸奎宁, 腹腔液, 鞘膜积液, 口形, 酰胺剂, 量＞, 应水化, 美斯纳, 1～, 融合肾, 12D, 骶骨, 依照, 资源, 行政性, 310, ATN, AGN多, ATN, 尿钠, 助于, ATN, 镓, 幼虫, 泌酸, 胆碱能神经, 6p21.2, 15q14, 8q24, 重构, 袢利尿剂, 醛固酮拮抗剂, 噻嗪类, 氯噻嗪, 美托拉宗, 袢利尿剂, 醛固酮拮抗剂, 利尿效果, 住院率, 尿中钾离子, 氯噻嗪利尿, 美托拉宗, 噻嗪类, 水潴留, 低血容量, 袢利尿剂, 噻嗪类, 高尿酸血症, 心外科, 班子, 单极, Schwann细胞, 琥珀酸, SDH, 琥珀酸, 不成熟, 电子显微镜, 甲肝病毒, 负染, 专门, WD病, 开拓, WD病, 导入, 高血磷, 结合剂, 钙盐, 碳酸钙, 乳酸钙, 双氢速固醇, 2.75, 满月, 睑裂, 赘皮, 耳道, 变长, 腭弓, 卧床不起, 环磷腺苷, NE, 递质, 脑区, 递质, 氨茶碱片, 小儿肿瘤学会, Wilms, 细胞组织学, favorable, histology, 多囊型, 纤维瘤样, unfavorable, histology, UH, 间变型, 透明细胞肉瘤, 细胞肉瘤, 出前, 较差, 甲基化, 四项, Biox, Ochia, Biox, Ochia, 脑白质区, 身心健康, 沉重, 前景, 长征, 注射液, N-乙酰酪氨酸, 氮源, 15-AA-823, 异常率, 门冬酰胺, 鸟氨酸, 赖氨酸, 组氨酸, 前清蛋白, 氮平衡, 纵观, 氮源, Y-V, 3.3, 15400, 均微, 止吐, 碱变性, Apt, 儿血, 氢氧化钠, 棕黄色, FHb, 抗碱, 高甘油三酯血症, 颈内动脉, 锁骨下动脉, 椎动脉, 全脑, 阴转率, 96%, 美洲, 涂搽, 美洲, 瑞萍, 1977, 1983, 年间, 自治区, 84, 901, 北部, 南部, 1957年, 79, 自治区, 北部, 黑龙江省, 哈尔滨市, 降至, 821, 泵泌, myocarditis, 心肌细胞, 文, 稀米汤样, 立方毫米, 数万, 每升, 数亿, 糖定量, 强阳性, 室性心动过速, tachycardia, 多克隆源性, 野生, 分枝, 菌菌, 列具, 无意义, 剔除, 细节, 蛲虫卵, 蛲虫病, 玻璃纸, 胶带, 紧压, 阀门, autohaler, 吸到, 贮雾器, spacer, 喷, 贮雾器, 贮雾器, 喷药, 吸药, 给药器, 静电, 吸粘, 器壁, 需予, 洗洁净, 贮雾器, 静电, 不锈钢, 贮雾器, group, streptococcic, streptococcus, pyogenes, 革兰阳性致病菌, 筋膜炎, 铸今, 消化吸收, 还原糖, 还原糖, 改良斑氏试剂, Clinitest, 比色, 赵, 祥文, 房性期前收缩, 结性期, 室性期前收缩, 阵发性室性心动过速, 室上性心动过速, 窦性心动过缓, 窦性停搏, 胆酸, 腔道, 口角炎, SRL+CsA+Pred, SRL+Pred, SRL+CsA+Pred, 专门, 小号, Sharp, 同室, 同室, 话语, 袖带, 充气囊, 周长, 周长, 袖带, 卵泡, HIV-1, HIV-2, HIV-1, 遍布, HIV-2, 西非, HIV-1, HIV-1, 阻滞药, 肌松药, 妥当, 呼吸波, 波长, 球样, 赘, 花样, 脆, 运, 单层柱状上皮, 脉络丛乳头状癌, papilloma, choroideum, 浸润性, 异形性, 良, 皆因, 四脑室肿瘤, 共济失调症, 侧脑室肿瘤, 偏盲, 脑血管造影, 血管造影, 脉络丛乳头状瘤, 脉络丛乳头状癌, 脉络丛乳头状癌, 也有, 可减少, 全切者, 12例, 5年, 66.6%, K<sub>1</sub>, 氨甲苯酸, PAMBA, 分出, 交通性脑积水, 阻塞性脑积水, 阻塞性脑积水, 占位性病变, 后遗, cytomegalovirus, 人巨细胞病毒, cytomegalovirus, 环指, 指腹, 摸到, 叫声, 清嗓声, 语词声, 模仿言语, 凉, 尿少, 多房性, 右室双出口, 关闭率, 艾森门格综合征, 活到, 血栓栓塞, 个人史, 过去史, 社会史, 七个, 抗癌药, 指点, 推给, 短句, 短句, 形容词, 脏, 短句, 所属, 预, 有的放矢, 咳喘, 和透皮吸收, 有的是, 数次, 不复存在, intussusception, 1674年, 荷兰人, Paul, Barbette, 1873年, Jonathan, Hutchinson, 1876年, Harlad, Hirschsprung, 复位法, Ravitch, 复位法, 1959年, 堪称, 源出, 词语类, 操作类, 词语类, 常识, 算术, 木块, 图案, 译码, 迷宫, 扑蛲灵, 渗透率, 后裔, Berger病, 激活性, 突眼, 甲状腺刺激免疫球蛋白, 雪莲, ∶114, 散布, 欧氏, node, 蜘蛛膜, 微氧, UTO, AD, nephrophthiasis, 胱氨酸病, 高血钙, 褪变, 肾小管基膜, 尿酸盐, 不定形, 尿酸盐, 结晶沉, 595, 773, 郎飞结, 尿胱氨酸, 高效液相色谱法, 曲霉菌球, 曲霉菌病, 肺部/n, 曲霉菌病, 外营养, parenteral, nutrition, 静脉高营养, intravenous, hyperalimentation, Dudrick, 正氮平衡, 南京, 造血干细胞, 扇动, 血管征, 透明膜病, inspiration, expiration, 失钾性, 推断, 钙质, 钙元素, 虾皮, 豆, 来潮, 瘦肉, 11.5, 海产品, 瘦肉, 隐患, 打下, 脑脊髓炎, 脑脊髓炎, 浅静脉, 套管, 钢针, 新华医院, 肠外营养, 慢速, 年龄越小, 引用, 外伤史, 激素类, 尼多酸钠, 酮替酚, 血碱性, 血甲状旁腺, l, 草莓状血管瘤, 海绵状, 混合瘤, 微静脉, 血管丛, 多层化基膜, 葡萄酒色斑, 橙红色斑, 海绵状血管瘤, 蔓状血管瘤, 超微, 胞质内, 内质网, ELISA法, 测血, 清抗, Western, blot, 应测, CP, 游戏化, 假单孢菌, 头孢类, 头孢哌酮, 尤需, 羟氨苄, 舒巴坦, 脑幕, 脑幕, 细小桥, 大脑半球, 等大, 弛缓性, 腰背, 细胞数, 颈蹼, 颈蹼, Tuner综合征, Noonan综合征, 胸锁乳突肌, 双肩部, 重症医学分会, 706, 706, 哽噎, 可诉, 样影, 取下, 后在, 刮取, 内生肌酐清除率, 尿醛固酮, 尿儿茶酚胺, 香草苦杏仁酸, 肾实质性高血压, 内分泌系, 神经系, 心血管系, 主动脉缩窄, 原发性醛固酮增多症, RVH者, 截面积, 音域, 电测听, 耳蜗, 背诵式, 多导, 戒烟, 戒烟, 毅力, 克-雅病, Creutzfeldt-Jakob, CJD, 接种人, 尾侧, 横结肠, SAM, 尾侧, 而来, 疝囊, B族溶血性链球菌, 北京市, 妇产, 1037, 11.07%, 9.95%, 14.92%, 李斯特菌, 滕, 晓唬, 其一, 其二, 贫穷, 镰状细胞, 肾外, 原发型, 周建华, 膀胱憩室, 神经结, 郎飞结, 发抖, 兄弟姐妹, 其为, 因子裂解蛋白酶, cleaving, protease, vWF-CP, ADAMTS-13, disintegrin, metalloproteinase, thrombospondin, motif, member, vWF多聚体, 调节因子, 补体因子H, complement, CFH, 补体因子Ⅰ, complement, factorⅠ, CFI, 补体膜辅助蛋白, cofactor, MCP, 微石症, microlithiasis, 52, 土耳其, 定类, 免疫复合物病, 令人满意, 室内压, 频度, 传到, 传到, 传到, 农业, 饲养, 磷酸二酯, phosphodiesterase, 鸟苷酸, 发挥作用, 磷酸二酯, PDE5, 磷酸二酯, 磷酸二酯, 肝胆, 布鲁菌病, 鼠咬热, 微需氧葡萄球菌, 小脓肿, 需氧链球菌, 微需氧链球菌, β溶血性, 非链球菌, 柯萨奇病毒B, TTF-Ⅰ, TTF-Ⅱ, Pax8, TSH-R, NIS, TG, TPO, 高频性, FSHD, FSHD, 房性心律, 用车, 三线, 该车, 箱, 箱, 滑动, 拆, 备, 驾驶室, 照明, 范可尼综合征, 病毒基因, 盖瑟尔, Gesell, 顶骨, 窦汇, 产钳, 坐骨, 上矢状窦, 泌尿系结核, 继发肾结核, 膀胱结核, 附睾结核, 仅供参考, 污染率, 前景, Vuori, Holopainen, 比拟, 良, 脆, 多数性, 慢性病变者, 黏膜桥, 隐窝炎, 上皮增生, 固有膜, 硬肿病, 诱聚剂, 醋酸盐, 碳酸盐, 电导率, 14.5, mS, 肿瘤者, 脑室枕角, 上视, Frank, Starling, 每搏量, 窃血, 亡, Fick, 胆管炎, 0.57%, 3.11%, 庆娅, 威, 黎华, 108, 720, 时间比, 封底, 彩图, 1d, 支持疗法, β肽链, 感染学, 非阿片类, 非阿片类, Reye综合征, 非阿片类, 天花板, 非阿片类镇痛剂, 喉返, 全层, 中膜, 血管弹力纤维, 甲皱, 甲皱, 右心缘, 抬, 叶间, 斜裂, AD, 常染色体显性遗传, X-连锁遗传, 发育不良, 关闭不全, 氧合作用, 依赖于, 气管镜, 沙门菌株, 病情严重, 情况不明, 噻肟等, UA, 上式, 移项, UA, <sup>-/w<, 五日, 风, 尖型, CCMD, Syndrome, 发病史, gllop, rhythm, 叉, 夹角, 夹角, 落入, 气陷, 补偿, 气道插管, 保暖箱, 发出报警, 盖, 数层, 布, 褥垫, 小流量, 血淋巴细胞, 丽, 华松, 惠珍, 非典型肺炎, 附6, 188, 188, pneumococcal, 铁锈色, 球状带, 状带, 中间层, 酮, 诸类, 羟, cytochrome, 超家族, 低密度, 条件者, 脱敏疗法, FMR, 发生变化, 传至, 时则, 传至, 时则, 缩减, 所携有, FMR, X染色体, 突变型, FMR, 突变型, FMR, 变化趋势, 缩减, 尚受, FMR, FMR, 定型化, 抽象思维, 性别差异, 流向, 流向, 静力, 流向, 剥, 叶间, 误伤, 粟粒, 积德, 122, 122, 保管, trachomatis, 滤泡性, 睑内翻, 睫毛, 经常性, 测者, 测处, 直角, 化学品, 可意料者, 消炎, 止痛药, 鸡胚羊膜腔, 阴转率, 泻药, 吐蛔虫, 碘酪氨酸, 侵扰, 侵害, 蛲虫病, enterobisis, 树芽, 珠样, 固有层, 相伴, 痒, 秃发, 脱发, 宽窄, 矮胖型, 封底, 彩图, 向内, 瘦长形, 封底, 彩图, 豚鼠, 荧光法, 阿拉斯加, 调查报告, 62, Ziegler, 84, 爱阿华州, 结果显示, 补充剂, 抗HBsAg, 酶标, 缩微版, 开具, 开具, 世界大战, 士兵们, 那时, 神经症, 那时, 战争, 平时期, 应激物, traumatic, stress, reserve, ERV, 闭陷, 谷草, 谷丙转氨酶, 几型, 颈面型放线菌, 胸部型放线菌, 腹部型放线菌, 皮肤型放线菌, 脑型放线菌, 放线菌性, 足菌病, 查血沉, 外斐, 抗溶血素链球菌, 精氨酸琥珀酸, 分别为, 515, 从事, 从中, 得益, obstructive, 其实不然, 打鼾, 打鼾, 过去史, 前角, 中继站, 初级, 葡萄簇型, 粗暴, 肤色, 抽泣, 30秒, 疫情报告, 牛乳钙, 后下部, 冠状窦口, 右移位, Yokoyama, 脑叶, 脑叶, 篇幅, 细述, 简述, 甘油盐水, 积分子, 铁丝, 圈样, 核碎片, 苏木素, 病变者, 五年, 检查台, 台边, 疑诊, ABO血型, ABO血型, AB型, 析出, 摄水量, 尿胱氨酸, 应妩, 规程, 南京, 东南大学, 725, 内化, 谈论, 自由, 公正, 价值观, 张嘴, 含氯, 熏蒸, 有效氯, m<sup>3</sup>, 含氯, 开门, streptococcal, pharyngitis, 磷脂酰甘油, 吸出液, 微泡计数法, 清洗液, 气泡, 大气泡, 排泄率, 指着, 数3, 4件, 从头, 报出, 数1, 后来, 相加, 相加, 库, 盛行, MUD, 移除, MUD, 移植者, 菲薄, 菲薄, 毛细毛细血管瘤, 手背, 鼻腔黏膜, 指压, 电凝法, immune, 异己, 一般而言, 变态反应性, vibration, duodenalulcer, gastriculcer, 胃穿孔, 间区, 同胃镜, 遭遇, 目睹, 杀人, 战斗, 肺总量, TLC, VC, conventional, mechanical, 兼用型, ，也, PCV, ，也, assist, 带动, 道正压, patient, triggered, PTV, assist, intermittent, synchronised, intermittent, intermittent, 示意, 同一种, 独特性, 有别于, 其他型, 链反应, 温抗, AIHA者, 植入体, 食道, Mahan, 外周动脉, 矢状窦, <sub>4/m<, 转铁蛋白, 楊, 丙昂, 执民, 2.24, 症候群, sulfasalazine, 神经体液, 耽搁, 胃隐窝, 肾漏型, 普遍认为, α1-AT, 蛋白溶解酶, α1-AT, 蛋白溶解酶, 肺组织蛋白, CD3, 后仰, 肾窝, 肾下垂, 肾下垂, 从来, 大概, 广泛应用, 围生, 多囊性, 发育不良, 多系, 肾下, 跨于, 之上, 男女比例, 一家, 几代, 尚待, CCMD, 垂体激素, 24h, 流应, 反流液, 数厘米, 一段, 面呈, 针尖状, 点影, 来回, 艳萍, 娟, .1988, 324, 324, 三核苷酸, 52, 精通, 主治医师, 麻醉科, 监护室, 本科, 麻醉科, 普外科, 放射科, 轮转, 毕业, 专业化, 床位, 护士长, 护师, 护士长, 职责, 运转, 配有, 工程师, 维修, 保养, 姊妹, 姊妹, 等臂, i, 荧光法, 交界线, 荧光带, 小钡条胶囊, 通过时间, 胜利, 给氧法, 肝胆系统, 硫胺素, 核对, 奚, 容平, cagA, vacA, 提问, 问答, 好奇性, 谈论, 果, 外在, 动物园, 脑出, 动静脉畸形, 多棘, 耳源性, 穿通, 最适, 温抗体, 冷抗体, IgG<sub>1</sub>, IgG<sub>3</sub>, IgG<sub>2</sub>, IgG<sub>4</sub>, 最适, 最适, 亚临床型, 乙患者, 血浆中因子, 炎型, 不符, 布鲁菌病, 肺吸虫, 相蛋白, 乳酸盐, 因子a, 子区, 598, 602, 尽少, 当事, 知情, 鼠伤寒沙门菌肠炎, propranolol, 氧需量, 阿替洛尔, atenolol, 美托洛尔, metoprolol, 品, 排泄性, 纵轴, 相交, 总阴, undetermined, anion, UA, 无机硫, 无机磷, 间液, 细胞内液, 细胞内液, 胶质错构瘤, 星形细胞瘤, 非钙化, 半透明肿瘤, 桑葚样, 色素缺失斑, 小眼球, 突眼, 磷酸吡哆醛, 少女, 两端, missense, mutation, 红细胞增多症, UA, 阳, UA-UC, 阳, 拿走, 钱物, 偷别, 外出, 行窃, 行窃, 行窃, 违法, 总容量, 香精, 薄荷, 氨水, 胡椒粉, pharmacokinetics, 药效学, pharmacodynamics, 小肠结肠炎, 马富西, 萎靡不振, 快感, 小儿胰腺炎, 1个, 消散性, 胆管炎, 诺如病毒, 血尿素, 血尿素, 柏油样, 停留, 停留, 清除术, 限局性, 测温, 肛表, 监护室, 综合性, 前进方向, 探及, 由上而下, 缓解者, 刮取物, 抽吸液, 查多克, Chaddock, 专心, 听讲, 做事, 虎头蛇尾, 拖拉, 动机, 反义核苷, 相容, 规范性, 大室, 单纯房, 红细胞增多症, 收缩因子, 亚临床型, 眼角, 入脑量, 首量, 注速, 紫癜肾炎, 第5对, 5<sub>P</sub>14～5<sub>P</sub>15, 第5号, 5<sub>P<, sub>-, 0.01%, Peterson, Stakey, ∶1, 庆云, 0.242%, 组化, isolated, 膨出壁, 低密度, 蛇头样影, 蛇头样, 磷脂类, 堵闭, 调节性, CD4<sup>+</sup>CD25<sup>+</sup>, 单细胞, 滑膜成纤维, 瀑布, 状体炎, 请, 弛缓性, 软瘫, 时肺, 内动, CcO2, CaO2, CcO2, CvO2, 可吸, 简略, 双氢克尿塞, 肝功, 抚养者, 清晰度, 出生史, 家庭史, 伸面, 腰椎脊, 铁一, 科学技术, 横结肠, 横结肠, 截断征, 轮廓不清, 肝肿大, 酮, 心肌灌注扫描, 心肌灌注不良, 冠状血管, 视乳头水肿, 较长, 主征, 四类, bFGF, 幼儿型, 染色体显性, 泌尿外科, 脓肿液, 癌瘤
```

## Performance

### Using Gold CWS (from POS file) and Test the Pure Mapping Model

> Fixed the `<sub> <sup>` tag problem, but the weird thing is all three value are the same. But if the CWS task is completely the same, maybe this is possible.

| Precision | Recall | F1 Score |
| --------- | ------ | -------- |
| 95.83     | 95.83  | 95.83    |

* [Precision and recall are equal when the size is same](https://stats.stackexchange.com/questions/97412/precision-and-recall-are-equal-when-the-size-is-same)

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

BiLSTM

* [**EricLingRui/NLP-tools**](https://github.com/EricLingRui/NLP-tools)

### Others

#### Application

English

* [Parts-of-speech.Info](https://parts-of-speech.info/)
* [Stanford Log-linear Part-Of-Speech Tagger](https://nlp.stanford.edu/software/tagger.html)

#### Precision and Recall

* [Precision vs Recall](https://towardsdatascience.com/precision-vs-recall-386cf9f89488)
* [Classification: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
