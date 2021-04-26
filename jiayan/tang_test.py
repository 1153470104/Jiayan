from jiayan import PMIEntropyLexiconConstructor
from jiayan import CharHMMTokenizer
from jiayan import TangCharHMMTokenizer
from jiayan import WordNgramTokenizer
from jiayan import CRFSentencizer
from jiayan import CRFPunctuator
from jiayan import CRFPOSTagger
from jiayan import load_lm


def hmm_tokenize(lm_path: str, text: str):
    lm = load_lm(lm_path)
    tokenizer = CharHMMTokenizer(lm)
    print(list(tokenizer.tokenize(text)))


def tang_tokenize(lm_path: str, text: str):
    lm = load_lm(lm_path)
    tokenizer = TangCharHMMTokenizer(lm)
    print(list(tokenizer.tokenize(text)))


if __name__ == '__main__':
    test = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也' \
           '判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方' \
           '悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'

    lm_path = 'C:/TJlab/Tang/chinese_poetry/jiayan.klm'

    print('\nTokenizing test text with HMM...')
    hmm_tokenize(lm_path, test)
