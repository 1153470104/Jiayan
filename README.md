# 甲言Jiayan
中文  
English  

## 简介
甲言，取「甲骨文言」之意，是一款专注于古汉语处理的NLP工具包。  
目前通用的汉语NLP工具均以现代汉语为核心语料，对古代汉语的处理效果很差(详见[分词](#__分词__))。本项目的初衷，便是辅助古汉语信息处理，帮助有志于挖掘古文化矿藏的古汉语学者、爱好者等更好地分析和利用文言资料，从「文化遗产」中创造出「文化新产」。  
当前版本支持词库构建、自动分词、文言句读和标点（目前仍有大量古籍未做句读）四项功能，更多功能正在开发中。  

Jiayan, which means Chinese characters engraved on the oracle bones, is an NLP toolkit designed for Classical Chinese.  

## 安装  
PyPI发布构建中。  
## 功能  
* [__词库构建__](#__词库构建__)  
  * 利用无监督的双[字典树](https://baike.baidu.com/item/Trie树)、[点互信息](https://www.jianshu.com/p/79de56cbb2c7)以及左右邻接[熵](https://baike.baidu.com/item/信息熵/7302318?fr=aladdin)进行文言词库自动构建。
* [__分词__](#__分词__)  
  * 利用无监督、无词典的[N元语法](https://baike.baidu.com/item/n元语法)和[隐马尔可夫模型](https://baike.baidu.com/item/隐马尔可夫模型)进行古汉语自动分词。
  * 利用词库构建功能产生的文言词典，基于有向无环词图、句子最大概率路径和动态规划算法进行分词。
* 词性标注  
  * 开发中，难点：
    1. 没有公开的古汉语词性标注语料；
    2. 现代汉语词性标注不适用于古汉语词；
    3. 难以直接使用无监督方法进行标注；
* [__断句__](#__断句__)
  * 基于字符的层叠式[条件随机场](https://baike.baidu.com/item/条件随机场)的序列标注，引入点互信息及[t-测试值](https://baike.baidu.com/item/t检验/9910799?fr=aladdin)为特征，对文言段落进行自动断句。
* [__标点__](#__标点__)
  * 基于字符的层叠式条件随机场的序列标注，在断句的基础上对文言段落进行自动标点。
* 文白翻译
  * 开发中，目前处于文白平行语料收集、清洗阶段。
  * 基于[双向长短时记忆循环网络](https://baike.baidu.com/item/长短期记忆人工神经网络/17541107?fromtitle=LSTM&fromid=17541102&fr=aladdin)和[注意力机制](https://baike.baidu.com/item/注意力机制)的神经网络生成模型，对古文进行自动翻译。
* 注意：受语料影响，目前不支持繁体。如需处理繁体，可先用[OpenCC](https://github.com/yichen0831/opencc-python)将输入转换为简体，再将结果转化为相应繁体(如港澳台等)。  
## 使用  
以下各模块的使用方法均来自[examples.py](jiayan/examples.py)。
1. 下载模型并解压：[百度网盘](https://pan.baidu.com/s/1N815EO8aEBl-S8Rd-rwXwA)，提取码：12gu
   * jiayan.klm：语言模型，主要用来分词，以及句读标点任务中的特征提取；  
   * cut_model：CRF句读模型；
   * punc_model：CRF标点模型；
   * 庄子.txt：用来测试词库构建的庄子全文。
2. __词库构建__  
   ```
   from jiayan import PMIEntropyLexiconConstructor
   
   constructor = PMIEntropyLexiconConstructor()
   lexicon = constructor.construct_lexicon('庄子.txt')
   constructor.save(lexicon, '庄子词库.csv')
   ```
   
   结果：  
   ```
   Word,Frequency,PMI,R_Entropy,L_Entropy
   之,2999,80,7.944909328101839,8.279435615456894
   而,2089,80,7.354575005231323,8.615211168836439
   不,1941,80,7.244331150611089,6.362131306822925
   ...
   天下,280,195.23602384978196,5.158574399464853,5.24731990592901
   圣人,111,150.0620531154239,4.622606551534004,4.6853474419338585
   万物,94,377.59805590304126,4.5959107835319895,4.538837960294887
   天地,92,186.73504238078462,3.1492586603863617,4.894533538722486
   孔子,80,176.2550051738876,4.284638190120882,2.4056390622295662
   庄子,76,169.26227942514097,2.328252899085616,2.1920058354921066
   仁义,58,882.3468468468468,3.501609497059026,4.96900162987599
   老聃,45,2281.2228260869565,2.384853500510039,2.4331958387289765
   ...
   ```
3. __分词__  
    1. 字符级隐马尔可夫模型分词，效果符合语感，建议使用，需加载语言模型 `jiayan.klm`
        ```
        from jiayan import load_lm
        from jiayan import CharHMMTokenizer
        
        text = '是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。'
        
        lm = load_lm('jiayan.klm')
        tokenizer = CharHMMTokenizer(lm)
        print(list(tokenizer.tokenize(text)))
        ```
        结果：  
        `['是', '故', '内圣外王', '之', '道', '，', '暗', '而', '不', '明', '，', '郁', '而', '不', '发', '，', '天下', '之', '人', '各', '为', '其', '所', '欲', '焉', '以', '自', '为', '方', '。']`  
        
        由于古汉语没有公开分词数据，无法做效果评估，但我们可以通过不同NLP工具对相同句子的处理结果来直观感受本项目的优势:  
        
        试比较 [LTP](https://github.com/HIT-SCIR/ltp) (3.4.0) 模型分词结果：  
        `['是', '故内', '圣外王', '之', '道', '，', '暗而不明', '，', '郁', '而', '不', '发', '，', '天下', '之', '人', '各', '为', '其', '所', '欲', '焉以自为方', '。']`  
        
        再试比较 [HanLP](http://hanlp.com) 分词结果：  
        `['是故', '内', '圣', '外', '王之道', '，', '暗', '而', '不明', '，', '郁', '而', '不', '发', '，', '天下', '之', '人', '各为其所欲焉', '以', '自为', '方', '。']`  
        
        可见本工具对古汉语的分词效果明显优于通用汉语NLP工具。  
        
    2. 词级最大概率路径分词，基本以字为单位，颗粒度较粗
        ```
        from jiayan import WordNgramTokenizer
        
        text = '是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。'
        tokenizer = WordNgramTokenizer()
        print(list(tokenizer.tokenize(text)))
        ```
        结果：  
        `['是', '故', '内', '圣', '外', '王', '之', '道', '，', '暗', '而', '不', '明', '，', '郁', '而', '不', '发', '，', '天下', '之', '人', '各', '为', '其', '所', '欲', '焉', '以', '自', '为', '方', '。']`  
4. __断句__
    ```
    from jiayan import load_lm
    from jiayan import CRFSentencizer
    
    text = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'
    
    lm = load_lm('jiayan.klm')
    sentencizer = CRFSentencizer(lm)
    sentencizer.load('cut_model')
    print(sentencizer.sentencize(text))
    ```
    结果：  
    `['天下大乱', '贤圣不明', '道德不一', '天下多得一察焉以自好', '譬如耳目', '皆有所明', '不能相通', '犹百家众技也', '皆有所长', '时有所用', '虽然', '不该不遍', '一之士也', '判天地之美', '析万物之理', '察古人之全', '寡能备于天地之美', '称神之容', '是故内圣外王之道', '暗而不明', '郁而不发', '天下之人各为其所欲焉以自为方', '悲夫', '百家往而不反', '必不合矣', '后世之学者', '不幸不见天地之纯', '古之大体', '道术将为天下裂']`  
5. __标点__
    ```
    from jiayan import load_lm
    from jiayan import CRFPunctuator
    
    text = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'
    
    lm = load_lm('jiayan.klm')
    punctuator = CRFPunctuator(lm, 'cut_model')
    punctuator.load('punc_model')
    print(punctuator.punctuate(text))
    ```
    结果：  
    `天下大乱，贤圣不明，道德不一，天下多得一察焉以自好，譬如耳目，皆有所明，不能相通，犹百家众技也，皆有所长，时有所用，虽然，不该不遍，一之士也，判天地之美，析万物之理，察古人之全，寡能备于天地之美，称神之容，是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方，悲夫！百家往而不反，必不合矣，后世之学者，不幸不见天地之纯，古之大体，道术将为天下裂。`

## 版本

