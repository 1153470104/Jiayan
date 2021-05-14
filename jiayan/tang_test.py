from jiayan import CharHMMTokenizer
from jiayan import TangCharHMMTokenizer
from jiayan import load_lm


def init_file():
    f1 = open("resource/qujiang_hmm.txt", 'w')
    f2 = open("resource/qujiang_tang.txt", 'w')
    f1.close()
    f2.close()


def list_to_file(path, p_list):
    f = open(path, 'a', encoding='utf-8')
    txt = ""
    for p in p_list:
        txt = txt+ p + " "
    txt = txt[:-1]
    f.write(txt)
    f.write("\n")
    f.close()


def list_to_text(p_list):
    txt = ""
    for p in p_list:
        txt = txt+ p + " "
    return txt[:-1]


if __name__ == '__main__':
    lm_path = 'C:/TJlab/Tang/chinese_poetry/jiayan.klm'
    print('\nTokenizing test text with HMM...')
    # init_file()
    lm = load_lm(lm_path)
    hmm_tokenizer = CharHMMTokenizer(lm)
    tang_tokenizer = TangCharHMMTokenizer(lm)

    # f = open("resource/qujiang_raw.txt", encoding='utf-8')
    # line = f.readline()
    # while line:
    #     # list_to_file("resource/qujiang_hmm.txt", list(tang_tokenizer.tokenize(line)))
    #     list_to_file("resource/qujiang_tang.txt", tang_tokenizer.intervene_tokenize(line))
    #     # list_to_file("resource/qujiang_tang_trans.txt", tang_tokenizer.intervene(line))
    #     line = f.readline()
    # f.close()

    text0 = "送春归，三月尽日日暮时。去年杏园花飞御沟绿，何处送春曲江曲。今年杜鹃花落子规啼，送春何处西江西。帝城送春犹怏怏" \
            "，天涯送春能不加惆怅。莫惆怅，送春人。冗员无替五年罢，应须准拟再送浔阳春。五年炎凉凡十变，又知此身健不健。" \
            "好去今年江上春，明年未死还相见。"
    text1 = "春泪烂罗绮，泣声抽恨多。莫滴芙蓉池，愁伤连蒂荷。壹朵花叶飞，壹枝花光彩。美人惜花心，但愿春长在。"
    text2 = "晦日同携手，临流壹望春。可怜杨柳陌，愁杀故乡人。"
    text3 = "去岁欢游何处去，曲江西岸杏园东。花下忘归因美景，尊前劝酒是春风。各从微宦风尘里，共度流年离别中。今日相逢愁又喜，八人分散两人同。"
    text4 = "及第新春选胜游，杏园初宴曲江头。紫毫粉壁题仙籍，柳色箫声拂御楼。霁景露光明远岸，晚空山翠坠芳洲。归时不省花间醉，绮陌香车似水流。"
    text5 = "寂寂孤莺啼杏园，寥寥壹犬吠桃源。落花芳草无寻处，万壑千峰独闭门。"

    # test
    # print(tang_tokenizer.seg_score("霁景"))
    # print(tang_tokenizer.seg_score("霁") + tang_tokenizer.seg_score("景"))
    # print(tang_tokenizer.seg_score("晚空"))
    # print(tang_tokenizer.seg_score("晚") + tang_tokenizer.seg_score("空"))

    # print(tang_tokenizer.validate(list(tang_tokenizer.sentences(text1))))

    # print(list_to_text(list(hmm_tokenizer.tokenize(text0))))
    # print(list_to_text(list(tang_tokenizer.tokenize(text0))))

    # print(list_to_text(list(hmm_tokenizer.tokenize(text1))))
    print(list_to_text(list(tang_tokenizer.tokenize(text1))))
    print(list_to_text(tang_tokenizer.intervene_tokenize(text1)))

    # print(list_to_text(list(hmm_tokenizer.tokenize(text2))))
    # print(list_to_text(list(tang_tokenizer.tokenize(text2))))
    # print(list_to_text(tang_tokenizer.intervene_tokenize(text2)))

    # print(list_to_text(list(hmm_tokenizer.tokenize(text3))))
    print(list_to_text(list(tang_tokenizer.tokenize(text3))))
    print(list_to_text(tang_tokenizer.intervene_tokenize(text3)))
    #
    # # print(list_to_text(list(hmm_tokenizer.tokenize(text4))))
    print(list_to_text(list(tang_tokenizer.tokenize(text4))))
    print(list_to_text(tang_tokenizer.intervene_tokenize(text4)))
    #
    # # print(list_to_text(list(hmm_tokenizer.tokenize(text5))))
    print(list_to_text(list(tang_tokenizer.tokenize(text5))))
    print(list_to_text(tang_tokenizer.intervene_tokenize(text5)))
