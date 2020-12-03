''' encoding=utf-8

Data processing Version 3.2

Date: 10/11/2020
Author: Yinya Huang

* argument words: pre-defined relations.
* domain words: repeated n-gram.


* relations patterns:
    1 - (relation, head, tail)  关键词在句首
    2 - (head, relation, tail)  关键词在句中，先因后果
    3 - (tail, relation, head)  关键词在句中，先果后因


== graph ==
    * undirected edges: periods.
    * directed edges: argument words.
    * nodes: chunks split by periods & argument words.


* change log of v3.0:
    1. remove text_norm() and _raw_tokenize()
    2. use BERT BPE tokens for argument word matching and domain word matching. (improving _find_arg_ngrams(),
        _find_dom_ngrams(), and _find_punct(), adapting the special Ġ in BPE tokens)

* change log of v3.1:
    1. double </s> between text_a and text_b -> single </s>. (Refer to tokenization_utils.py and tokenization_utils_base.py)

* change log of v3.2:
    1. remove preprocess()



'''

from dataclasses import dataclass, field
import argparse
from transformers import AutoTokenizer
import gensim
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")


# def preprocess(text, do_lower_case):
#     '''
#     1. correct the types.
#     2. lower.
#     '''
#     output = TextBlob(text).correct()
#     output = str(output)
#     if do_lower_case:
#         output = output.lower()
#     return output

# def text_norm(text):
#     doc = nlp(text)
#     lemmatized = [t.lemma_ if 'PRON' not in t.lemma_ else t.text for t in doc]
#     return " ".join(lemmatized)
#
# def token_norm(token):
#     doc = nlp(token)
#     lemmatized = [t.lemma_ if 'PRON' not in t.lemma_ else t.text for t in doc]
#     return "".join(lemmatized)

def token_stem(token):
    return stemmer.stem(token)

def find_space_bpe_mapping(space_token_list, bpe_token_list):
    # by rainyucao
    space_i, bpe_i = 0, 0
    res = []
    prev_text = ''
    tmp_mapping = []
    while space_i < len(space_token_list):
        prev_text = prev_text + bpe_token_list[bpe_i]
        tmp_mapping.append(bpe_i)
        if prev_text == space_token_list[space_i]:
            res.append(tmp_mapping)
            prev_text = ''
            space_i += 1
            tmp_mapping = []
        bpe_i += 1
    return res


def find_space_bpe_mapping_ids(space_token_list, bpe_token_list, space_id_list):
    assert len(space_token_list) == len(space_id_list)
    space_i, bpe_i = 0, 0
    res = []
    prev_text = ''
    tmp_mapping = []
    while space_i < len(space_token_list):
        prev_text = prev_text + bpe_token_list[bpe_i]
        tmp_mapping.append(space_id_list[space_i])
        if prev_text == space_token_list[space_i]:
            # res.append(tmp_mapping)
            res += tmp_mapping
            prev_text = ''
            space_i += 1
            tmp_mapping = []
        bpe_i += 1
    assert len(res) == len(bpe_token_list)
    return res


@dataclass(frozen=False)
class ArgumentToken:
    text: str
    space_window: tuple # (int, int)
    space_len: int
    is_stopwords: bool


def arg_tokenizer(text_a, text_b, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        # arg_flag = False
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            # arg_flag = True
            pattern = argument_words[seq.strip()]
        # return arg_flag
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids

    def _find_arg_ngrams(tokens, max_gram):
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end])
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        # argument_words.append(argument_token)
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)  # {1/2/3}
                        # argument_words[ngram] = tokens[window_start:window_end]
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 判断 stopwords 和 sep token
        2. 先遍历一遍，记录 n-gram 的重复次数和出现位置
        3. 遍历记录的 n-gram, 过滤掉 n-gram 子序列（直接比较 str）
        4. 赋值 domain_ids.

        Change log:
            - input tokens are BERT BPE tokens.
            - first norm input tokens.

        '''

        # normed_tokens = [token_norm(token) for token in tokens]  # failure case e.g., relaxed -> relaxed.
        stemmed_tokens = [token_stem(token) for token in tokens]
        # print('before after norm: {}'.format([(a,b,c) for a,b,c in zip(tokens, normed, stemmed)]))

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                # record.
                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        # print('tokens', tokens)
        # print('raw_domain_words', raw_domain_words)
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    # print('\"{}\" in \"{}\"'.format(raw_domain_words[i], raw_domain_words[j]))
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    # print('\"{}\" in \"{}\"'.format(raw_domain_words[j], raw_domain_words[i]))
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                # print('{} is not in domain_words_to_remove'.format(r))
                pass

        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]

        #
        # print('== token, arg_id ==')
        # for i, (token, id) in enumerate(zip(tokens, domain_ids)):
        #     print(i, token, id)
        # print('== dom_words ==')
        # for k,v in domain_words.items():
        #     print(k,v)
        # assert 1 == 0

        return domain_words_stemmed, domain_words_orin, domain_ids



    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    # bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
    #              [tokenizer.sep_token] + bpe_tokens_b + [tokenizer.eos_token]
    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.eos_token]

    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - len(bpe_tokens))
    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)

    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens]
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)  # checked.
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)  # checked.
    punct_space_ids = _find_punct(bare_tokens, punctuations)  # checked.

    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids
    punct_bpe_ids = punct_space_ids

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)

    # paddings.
    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    domain_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    input_mask += padding
    segment_ids += padding

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)

    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask

    return output


def main(text, option):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        # default='/data2/yinyahuang/BERT_MODELS/roberta-large-uncased'
        default='/Users/mac/Documents/__myprojects/BERT_MODELS/roberta-large-uncased'
        # default='/Users/mac/Documents/__Datasets/drop_dataset/roberta.large'
        )
    parser.add_argument(
        '--max_gram',
        type=int,
        default=5,
        help="max ngram for window sliding."
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations

    inputs = arg_tokenizer(text, option, tokenizer, stopwords, relations, punctuations,
                           args.max_gram, args.max_length)


    ''' print '''
    p = []
    for token, arg, dom, pun in zip(inputs["input_tokens"], inputs["argument_bpe_ids"], inputs["domain_bpe_ids"],
                                    inputs["punct_bpe_ids"]):
        p.append((token, arg, dom, pun))
    print(p)
    print('input_tokens\n{}'.format(inputs["input_tokens"]))
    print('input_ids\n{}, size={}'.format(inputs["input_ids"], len(inputs["input_ids"])))
    print('attention_mask\n{}'.format(inputs["attention_mask"]))
    print('token_type_ids\n{}'.format(inputs["token_type_ids"]))
    print('argument_bpe_ids\n{}'.format(inputs["argument_bpe_ids"]))
    print('domain_bpe_ids\n{}, size={}'.format(inputs["domain_bpe_ids"], len(inputs["domain_bpe_ids"])))
    print('punct_bpe_ids\n{}'.format(inputs["punct_bpe_ids"]))
    # print('a_mask\n{}'.format(inputs["a_mask"]))
    # print('b_mask\n{}'.format(inputs["b_mask"]))


if __name__ == '__main__':

    import json
    from graph_building_blocks.argument_set_punctuation_v4 import punctuations
    with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
        relations = json.load(f)  # key: relations, value: ignore

    # text = 'The position that punishment should be proportional to how serious the offense is but that repeat ' \
    #        'offenders should receive harsher punishments than first-time offenders is unsustainable. ' \
    #        'It implies that considerations as remote as what an offender did years ago are relevant ' \
    #        'to the seriousness of an offense. If such remote considerations were relevant, ' \
    #        'almost every other consideration would be too. But this would make determining ' \
    #        'the seriousness of an offense so difficult that it would be impossible to apply ' \
    #        'the proportionality principle.'

    # text = 'A recent study monitored the blood pressure of people petting domestic animals in the laboratory. ' \
    #        'The blood pressure of some of these people lowered while petting the animals. Ttherefore, ' \
    #        'for any one of the people so affected, owning a pet would result in that person having a lower average ' \
    #        'blood pressure.'
    #
    # option = 'Since riding in a boat for a few minutes is relaxing for some people, ' \
    #          'those people would be more relaxed generally if those people owned boats.'

    # text = 'Recently, a report commissioned by a confectioners trade association noted that chocolate, formerly ' \
    #        'considered a health scourge, is an effective antioxidant and so has health benefits. Another earlier ' \
    #        'claim was that oily foods clog arteries, leading to heart disease, yet reports now state that olive oil ' \
    #        'has a positive influence on the circulatory system. From these examples, it is clear that if you wait ' \
    #        'ong enough, almost any food will be reported to be healthful.'

    # option = 'Everyone with an office on the second floor works directly for the president and, as a result, no one ' \
    #          'with a second floor office will take a July vacation because no one who works for the president will ' \
    #          'be able to take time off during July.'

    # text = 'In Rubaria, excellent health care is available to virtually the entire population, whereas very few ' \
    #        'people in Terland receive adequate medical care. Yet, although the death rate for most diseases is ' \
    #        'higher in Terland than in Rubaria, the percentage of the male population that dies from prostate cancer ' \
    #        'is significantly higher in Rubaria than in Terland.'

    # text = 'Rye sown in the fall and plowed into the soil in early spring leaves a residue that is highly effective ' \
    #        'at controlling broad-leaved weeds, but unfortunately for only about 45 days. No major agricultural crop ' \
           # 'matures from seed in as little as 45 days. Synthetic herbicides, on the other hand, although not any ' \
           # 'longer-lasting, can be reapplied as the crop grows. Clearly, ttherefore, for major agricultural crops, ' \
           # 'plowing rye into the soil can play no part in effective weed control.'

    # text = 'Economist: In free market systems, the primary responsibility of corporate executives is to determine ' \
    #        'a nation\' s industrial technology, the pattern of work organization, location of industry, and resource ' \
    #        'allocation. They also are the decision makers, though subject to significant consumer control, on what ' \
    #        'is to be produced and in what quantities. In short, a large category of major decisions is turned over ' \
    #        'to business executives. Thus, business executives have become public officials.'

    # text = '"The rate at which a road wears depends on various factors, including climate, amount of traffic, and ' \
    #        'the size and weight of the vehicles using it. The only land transportation to Rittland\'s seaport ' \
    #        'is via a divided highway, one side carrying traffic to the seaport and one carrying traffic away from ' \
    #        'it. The side leading to the seaport has worn faster, even though each side has carried virtually the ' \
    #        'same amount of traffic, consisting mainly of large trucks.'

    text = 'Until he was dismissed amid great controversy, Hastings was considered one of the greatest intelligence ' \
           'agents of all time. It is clear that if his dismissal was justified, then Hastings was either ' \
           'incompetent or else disloyal. Soon after the dismissal, however, it was shown that he had never been ' \
           'incompetent. Thus, one is forced to conclude that Hastings must have been disloyal.'

    option = 'Everyone with an office on the second floor works directly for the president and, as a result, no one ' \
             'with a second floor office will take a July vacation because no one who works for the president will ' \
             'be able to take time off during July.'

    # set_argument = {"turn"}
    # set_domain = {"pain", "normal", "medic", "cell", "injuri", "hormon", "protein", "reaction", "caus", "turn",
    #               "function", "bodi", "releas", "contain", "healthi", "attack", "activ", "immun", "infect"}

    main(text, option)