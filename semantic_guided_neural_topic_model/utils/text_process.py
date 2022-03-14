from semantic_guided_neural_topic_model.utils import resources_dir
from typing import Iterable
from os.path import join
import nltk.corpus.reader.wordnet as wn
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer

# lemmatizer
lemmatizer = WordNetLemmatizer()

# stopwords
stopwords = None
stopwords_path = join(resources_dir, "stopwords.txt")


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
    if len(token) <= 2:
        return token
    n_lemma = lemmatizer.lemmatize(token, pos=wn.NOUN)
    if n_lemma != token:
        return n_lemma
    v_lemma = lemmatizer.lemmatize(token, pos=wn.VERB)
    return v_lemma


def build_bow_vocab(texts: Iterable[Iterable[str]], no_below: int = 3, no_above: float = 0.5, keep_n: int = 100000,
                    keep_tokens: Iterable[str] = None, is_lemmatize=True) -> Dictionary:
    if is_lemmatize:
        dictionary = Dictionary((simple_lemmatize(word) for word in text if len(word.strip()) > 1) for text in texts)
    else:
        dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)
    dictionary.compactify()
    return dictionary


def get_stopwords():
    global stopwords
    if stopwords is None:
        with open(stopwords_path) as in_file:
            stopwords = [word.strip() for word in in_file]
        stopwords = frozenset(stopwords)
    return stopwords
