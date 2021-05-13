from math import log10

from jiayan.globals import re_zh_include, stopchars

"""
this program first reuses the implementation of tokenizer in jiayan
then does some modification based on the specific features of my corpus object: Tang poetry

唐诗语言结构: 唐诗主要分为五言与七言，五言采用2+2+1 或者2+1+2 的句法, 七言采用2+2+1+2 和2+2+2+1 的句法
分词结合诗歌结构: 在不考虑诗词结构时转移概率仅由语言模型决定，但考虑唐诗结构后转移概率可以与字在诗句中的位置挂钩：
规则考虑如下：
1. 五言:  前两个字的合并问题 1+1+2+1  ->  2+1+2   ;   1+1+1+2  ->  2+1+2 
          后三个字的合并问题 2+1+1+1  ->  2+1+2 / 2+2+1 (base on the last 3 words' prob)
          跨 2字 3字 的问题 1+2+1+1  -> 1+1+1+1
2. 七言:  前面两两成对合并问题 1+1+2+1+2  ->  2+2+1+2   ;   2+1+1+2+1  ->  2+2+2+1   ;   1+1+1+1+(1+2)  -> 2+2+(1+2)
          后三字合并问题 (2+2)+1+1+1  ->  (2+2)+2+1 / (2+2)+1+2
          跨 2 2 字、跨 4 3 字的问题 1+2+1+(1+2)  ->  1+1+1+1+(1+2)    ;   2+1+2+2  ->  2+1+1+1+2
3. 冲突:  前后两句理应对账，诗句的句法也应当基本相同
          因此如果出现句法不同的情况，视转移概率的大小，来判断是要拆还是合并
4. 长词: 出现长词时，即第三第四字时，考虑是否跨越边界而不给出转移概率了

* 单字的合并需要考虑：转移概率是否过低，甚至没有。如果低于某个阈值，则不予合并。

--------------------- previous documents written by jiayan's author ---------------------

Use HMM to consider word detection as a char sequence tagging problem.

With a word dict and a char sequence, there could be lots of tokenizing solutions, and the best one will have
the biggest multiplication probability of words:
(see Max Probability Tokenizing: [https://blog.csdn.net/u010189459/article/details/37956689])
p(S) = p(w1) * p(w2) * p(w3)...p(wn)

However, without a word dict we don't know how to tokenize the sentence by word. But here we can use
language model to compute a possible word probability first:
p(w) = p(c1, c2, c3, c4) = p(c1) * p(c2|c1) * p(c3|c1, c2) * p(c4|c1, c2, c3)

Here the word "w" is a 4-char word, with c1, c2, c3 and c4, and the probabilities of each char occurring in relative
position could be computed with N-grams model.

So assume the longest word we want is 4-char word, then in a sentence with length L (L char sequence), each char 
could be in 4 possible positions of one word, and each associates with its probability of being at that position
(k indicates the kth char in the sequence)

1. the beginning of the word (b): p(ck)
2. the second char of the word (c): p(ck|ck-1)
3. the third char of the word (d): p(ck|ck-2, ck-1)
4. the fourth char of the word (e): p(ck|ck-3, ck-2, ck-1)

So, a char sequence could be tagged in a char level with labels {b, c, d, e} first, and be chunked based on the
tags. Now we can see the word level problem is broken down to char level problem with hidden states, so this is the 
decoding problem of HMM, we can use viterbi algorithm to get the best tag/state sequence for the char/observation
sequence.

For viterbi, we need (a)initial starting probabilities of each state, (b)transition probabilities between states, and
(c)emission probabilities of states emitting different observations. Let's draw a table to see what they should be in
this problem:

----------------------------------------------------
   start      ->    b           b           b
                    c           c           c
                    d           d           d
                    e           e           e

char sequence:    char1       char2        char3 ...
-----------------------------------------------------

So for each char in the sequence, there are 4 possible states.
For (a), only "b" can start a sequence, so p(b|<S>) = 1, and p(c|<S>) = p(d|<S>) = p(e|<S>) = 0
For (b), consider the longest word: "bcde", we can see the state transitions are limited in:
    i.   b -> b, b -> c: the beginning of a word either goes to a new word beginning, or the 2nd char;
    ii.  c -> b, c -> d: the 2nd char either goes to a new word beginning, or the 3rd char;
    iii. d -> b, d -> e: the 3rd char either goes to a new word beginning, or the 4th char;
    iv.  e -> b, e -> e: the 4th char either goes to a new word beginning, or the 5th char ...
For (c), the emission probability of one char at a certain state could be computed with N-grams model, e.g.,
    p(ck|d) = p(ck|ck-1, ck-2)

The only parameters that we cannot compute here are transition probabilities, which we can manually set.

Differences from regular HMM tokenizing:
(a) regular HMM tokenizing uses label set {B, M, E, S} to tag char sequence, which is very vague to indicate
    exact char position within a word, especially "M", thus hard to compute emission probabilities;
(b) regular HMM tokenizing requires large data to compute transition and emission probabilities, but here our
    goal is the opposite, to generate that word corpus;
(c) regular HMM tokenizing computes transition probabilities from data, but here we set them manually;
(d) regular HMM tokenizing computes emission probabilities from data, but here we use char level N-grams 
    language model.

Disadvantages:
(a) slow: read the sentence data to build ngrams from min word length to max word length, and read again to tokenize
          the whole data, and by this to build word corpus; viterbi on each sentence in data
(b) bad at long word: need to fine tune transition probabilities to control the word lengths, which is time consuming,
          and the detected long words are not as good as short words.
(c) fake word frequency: since word corpus is built by tokenizing, which can lead to inaccurate sentence splits, the
          word count doesn't reflect true frequency, e.g., "天下" in "于天下". So we use its true frequency count in 
          the ngrams dict when filtering.
"""


class TangCharHMMTokenizer:

    def __init__(self, lm):
        self.lm = lm
        self.inits = {'b': 0.0, 'c': -3.14e100, 'd': -3.14e100, 'e': -3.14e100}

        # the transition probabilities are manually fine tuned;
        # in principle, we would like the word length the shorter the better;
        # low to-b and high to-next-char-in-word transition probs lead to long words;
        # high to-b and low to-next-char-in-word transition probs lead to short words.

        trans = {'bb': 0.9, 'bc': 0.1,
                 'cb': 0.996, 'cd': 0.004,
                 'db': 0.999, 'de': 0.001,
                 'eb': 0.9999, 'ee': 0.0001}

        # 以下的是jiayan作者指定的转移概率，未来可以继续修改
        # trans = {'bb': 0.85, 'bc': 0.15,
        #          'cb': 0.9925, 'cd': 0.0075,
        #          'db': 0.999, 'de': 0.001,
        #          'eb': 0.9999, 'ee': 0.0001}

        # trans = {'bb': 0.8, 'bc': 0.2,
        #          'cb': 0.9925, 'cd': 0.0075,
        #          'db': 0.999, 'de': 0.001,
        #          'eb': 0.9999, 'ee': 0.0001}

        # convert the decimal probabilities to logs to avoid overflow
        self.trans = {states: log10(trans_prob) for states, trans_prob in trans.items()}

    def tokenize(self, text: str):
        """ Gets the tags of given sentence, and tokenizes sentence based on the tag sequence.
        """
        # split text by whitespaces first, then split each segment into char chunks by zh chars
        for seg in text.strip().split():  # 如果split()没有参数意味着会以whitespace来分割
            if seg:
                for chunk in re_zh_include.split(seg):
                    # if zh chars, tokenize them
                    if re_zh_include.match(chunk):
                        tags = self.viterbi(chunk)

                        word = chunk[0]
                        for i in range(1, len(chunk)):
                            if tags[i] == 'b':
                                if not self.valid_word(word):
                                    for char in word:
                                        yield char
                                else:
                                    yield word
                                word = chunk[i]
                            else:
                                word += chunk[i]
                        if word:
                            if not self.valid_word(word):
                                for char in word:
                                    yield char
                            else:
                                yield word

                    # if not zh chars, we assume they are all punctuations, split them
                    else:
                        for char in chunk:
                            yield char

    def sentences(self, words):
        sentences = []
        temp = []
        for w in words:
            if w == '，' or w == '。' or w == ',':
                temp.append(w)
                sentences.append(temp)
                temp = []
            else:
                temp.append(w)
        if temp:
            sentences.append(temp)
        return sentences

    def validate(self, sentences):
        if len(sentences) % 2 != 0:
            return -1
        s1 = sentences[0]
        length = 0
        for w in s1:
            length = length+len(w)
        # print("length: "+length)
        for s in sentences:
            temp_length = 0
            for w in s:
                temp_length = temp_length+len(w)
            if temp_length != length:
                return -1
        return length - 1

    def intervene_tokenize(self, text: str):
        words = self.tokenize(text)
        sentences = self.sentences(list(words))  # 用于得到分句的分词结果。
        valid_num = self.validate(sentences)
        if valid_num == -1:
            return words
        else:
            self.segment_intervene(sentences, valid_num)
            self.confront_intervene(sentences, valid_num)

        new_words = []
        for s in sentences:
            for w in s:
                new_words.append(w)
        return new_words

    def segment_intervene(self, sentences, length):
        """ 用于不合理的拆分的判断与修改
        """
        # print(sentences)
        for s in sentences:
            pos = 0
            list_len = len(s)
            i = 0
            seven_two_mark = False
            while i < list_len:
                if length == 5:
                    '''五言句子3-2分割为必须'''
                    pos += len(s[i])
                    if pos == 2:
                        i += 1
                        break  # 这里先写了pass又换成continue，最后才发现是写break。。。
                    if pos >= 3:
                        w = s.pop(i)
                        s.insert(i, w[:2-pos])
                        i += 1
                        s.insert(i, w[2-pos:])
                        break  # 分割之后可以直接break都不需要干别的
                    i += 1  # 循环条件居然都写错。。。没有i++真的是生疏了，这才几天就生疏了

                elif length == 7:
                    ''' 七言句子4-3分割为必须 '''
                    pos += len(s[i])
                    if pos == 2:
                        seven_two_mark = True
                    if pos >= 3 and not seven_two_mark:
                        w = s.pop(i)
                        s.insert(i, w[:2-pos])
                        i += 1
                        s.insert(i, w[2-pos:])
                        seven_two_mark = True
                    if pos == 4 and seven_two_mark:
                        i += 1
                        break
                    if pos >= 5 and seven_two_mark:
                        w = s.pop(i)
                        s.insert(i, w[:4-pos])
                        i += 1
                        s.insert(i, w[4-pos:])
                        break
                    i += 1

    def confront_intervene(self, sentences, length):
        """ 用于对仗的判断与修改
            写的很丑陋。。。真是直接手写了，或许有更好的形式吧，先写出来再说
        """
        count = int(len(sentences) / 2)
        '''由于绝句不对仗，律诗大部分首联尾联不对仗，所以取1 ~ count-1'''
        for i in range(1, count-1):
            s1 = sentences[2 * i]
            s2 = sentences[2 * i + 1]
            news1 = []
            news2 = []
            if length == 5:
                p1 = self.compare(self.get_part(s1, 0, 5), self.get_part(s2, 0, 5), 2)
                p2 = self.compare(self.get_part(s1, 1, 5), self.get_part(s2, 1, 5), 3)
                news1.extend(p1[0])
                news1.extend(p2[0])
                news1.append(s1[-1])
                news2.extend(p1[1])
                news2.extend(p2[1])
                news2.append(s2[-1])
            elif length == 7:
                p1 = self.compare(self.get_part(s1, 0, 7), self.get_part(s2, 0, 7), 2)
                p2 = self.compare(self.get_part(s1, 1, 7), self.get_part(s2, 1, 7), 2)
                p3 = self.compare(self.get_part(s1, 2, 7), self.get_part(s2, 2, 7), 3)
                news1.extend(p1[0])
                news1.extend(p2[0])
                news1.extend(p3[0])
                news1.append(s1[-1])
                news2.extend(p1[1])
                news2.extend(p2[1])
                news2.extend(p3[1])
                news2.append(s2[-1])

    def compare(self, list1, list2, length):
        """ 接收两个字符列表，返回一个包含两个调整过的字符列表的大列表
        """
        ''' 这里目前只考虑的是一个分一个没分时合并的问题 '''
        result = []
        if length == 2:
            if len(list1) > len(list2):
                if self.seg_score(list1[0]) + self.seg_score(list1[1]) < self.seg_score(list1[0]+list1[1])+log10(0.2):
                    temp = list1[0]+list1[1]
                    list1 = [temp]
                    print(temp)
            elif len(list1) < len(list2):
                if self.seg_score(list2[0]) + self.seg_score(list2[1]) < self.seg_score(list2[0]+list2[1])+log10(0.2):
                    temp = list2[0]+list2[1]
                    list2 = [temp]
                    print(temp)
        elif length == 3:  # 实际上三个词的分解会涉及到地名被拆分，所以未来估计要加入词表，用来减少这种问题
            if len(list1) == 1:
                temp = list1[0]
                list1 = []
                count = 0
                for w in list2:
                    list1.append(temp[count: count+len(w)])
                    count += len(w)
                print(list1)
            elif len(list2) == 1:
                temp = list2[0]
                list2 = []
                count = 0
                for w in list1:
                    list2.append(temp[count: count+len(w)])
                    count += len(w)
                print(list2)

        result.append(list1)
        result.append(list2)

        return result

    def get_part(self, sentence, num, length):
        """ 这里得到的是诗句中某个部分的集合
              如果是五言则是2-3的某个部分
              如果是七言则是2-2-3的某个部分
            这里是否是五言七言由使用者控制！！！别写错了，因为不想写validate了。。。
        """
        start_index = 0
        end_index = 0
        count = 0
        if num == 0:
            start_index = 0
            end_index = 2
        elif num == 1 and length == 5:
            start_index = 2
            end_index = 5
        elif num == 1 and length == 7:
            start_index = 2
            end_index = 4  # 本来这里居然是5，写的时候贪图方便直接复制，铸成大错
        elif num == 3:
            start_index = 4
            end_index = 7

        result = []
        for w in sentence:
            count += len(w)
            if start_index < count <= end_index:
                result.append(w)
        return result

    def viterbi(self, sent):
        """ Chooses the most likely char tag sequence of given char sentence.
        """
        emits = self.get_emission_probs(sent)

        # record the best path for each state for each char, {path1: path_prob, path2: path_prob, ...};
        # paths grow at each decoding step, eventually contains the best paths for each state of last char;
        # we assume the initial state probs = 1st char's emission probs
        paths = {state: prob + self.inits[state] for state, prob in emits[0].items()}

        # for each char
        for i in range(1, len(sent)):
            # print(paths)

            # record best paths and their probs to all states of current char
            cur_char_paths = {}

            # for each state of current char
            for state, emit_prob in emits[i].items():

                # record all possible paths and their probs to current state
                cur_state_paths = {}

                # for each state of previous char
                for path, path_prob in paths.items():
                    trans_states = path[-1] + state

                    # compute the path prob from a previous state to current state
                    if trans_states in self.trans:
                        cur_state_paths[path + state] = path_prob + emit_prob + self.trans[trans_states]

                # choose the best path from all previous paths to current state
                best_path = sorted(cur_state_paths, key=lambda x: cur_state_paths[x])[-1]

                # for current state of current char, we found its best path
                cur_char_paths[best_path] = cur_state_paths[best_path]

            # the paths grow by one char/state
            paths = cur_char_paths

        return sorted(paths, key=lambda x: paths[x])[-1]

    def get_emission_probs(self, sent):
        """ Computes emission probability of each state emitting relative char in the given char sequence. """
        return [

            {'b': self.seg_prob(sent[i]),
             'c': self.seg_prob(sent[i - 1:i + 1]),
             'd': self.seg_prob(sent[i - 2:i + 1]),
             'e': self.seg_prob(sent[i - 3:i + 1])
             }

            for i in range(len(sent))
        ]

    def seg_score(self, seg):
        """ 这里仅仅获取这这个seg片段的 n-gram score 值
        """
        return self.lm.score(' '.join(seg), bos=False, eos=False)

    def seg_prob(self, seg):
        """ Computes the segment probability based on ngrams model.
            If given an empty segment, it means it's impossible for current char to be at current position of a word,
            thus return default low log prob -100.
        """
        return (self.lm.score(' '.join(seg), bos=False, eos=False) -
                self.lm.score(' '.join(seg[:-1]), bos=False, eos=False)) \
               or -100.0

    def valid_word(self, word):
        """ Checks if a word contains stopchars, if yes, it's not a valid word. """
        for char in word:
            if char in stopchars:
                return False
        return True
