from typing import Iterable

import nltk.corpus.reader.wordnet as wn
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def read_raw_vocab(full_path):
    """
    line format: word
    """
    id2token = {}
    token2id = {}
    with open(full_path, mode='r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            id2token[idx] = line
            token2id[line] = idx
    return id2token, token2id


def simple_lemmatize(token):
    token = token.strip()
    if token <= 2:
        return token
    n_lemma = lemmatizer.lemmatize(token, pos=wn.NOUN)
    if n_lemma != token:
        return n_lemma
    v_lemma = lemmatizer.lemmatize(token, pos=wn.VERB)
    return v_lemma


def build_bow_vocab(texts: Iterable[Iterable[str]], no_below: int = 3, no_above: float = 0.5, keep_n: int = 100000,
                    keep_tokens: Iterable[str] = None) -> Dictionary:
    dictionary = Dictionary((simple_lemmatize(word) for word in text if len(word.strip()) > 1) for text in texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)
    dictionary.compactify()
    return dictionary
