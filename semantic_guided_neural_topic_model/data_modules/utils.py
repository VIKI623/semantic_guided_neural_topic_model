from functools import partial
from typing import Mapping, Iterable, Optional, Sequence
import numpy as np
from datasets import load_dataset
from semantic_guided_neural_topic_model.pretrain_modules.models import SentenceBertModel
from semantic_guided_neural_topic_model.utils.text_process import read_raw_vocab, build_bow_vocab

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def filter_dataset_by_attribute_value_length(dataset, attribute_key='text_for_bow', no_below: int = 2):
    # filter
    dataset = dataset.filter(lambda sample: len(
        sample[attribute_key]) >= no_below)
    return dataset


def bow_to_clean_bow(text_for_bow: Iterable[str], token2id: Mapping[str, int]) -> Sequence[str]:
    return [word for word in text_for_bow if word in token2id]


def load_bow(raw_json_file: str = None, raw_vocab_file: str = None, normalization: Optional[str] = None, batch_size: int = 64):
    dataset = load_dataset('json', data_files=raw_json_file)['train']

    if raw_json_file:
        id2token, token2id = read_raw_vocab(raw_vocab_file)
    else:
        dictionary = build_bow_vocab(texts=dataset["text_for_bow"])
        id2token, token2id = dict(dictionary), dict(dictionary.token2id)

    # remove word not in dictionary
    bow_to_clean_bow_with_vocab = partial(bow_to_clean_bow, token2id=token2id)
    dataset = dataset.map(lambda item: {"text_for_bow": bow_to_clean_bow_with_vocab(item["text_for_bow"])},
                          batched=False)

    # filter by text_for_bow length
    dataset = filter_dataset_by_attribute_value_length(
        dataset, attribute_key="text_for_bow", no_below=2)

    # map text_for_bow to bow
    handlers = [('bow', CountVectorizer(vocabulary=token2id)), ]
    if normalization and normalization.lower() == 'tfidf':
        handlers.append(('tfidf', TfidfTransformer(norm='l1')))

    pipe = Pipeline(handlers)

    dataset = dataset.map(lambda items: {"bow": pipe.fit_transform(
        [' '.join(item) for item in items["text_for_bow"]]).astype(np.float32).toarray()}, batched=True, batch_size=batch_size)

    return dataset, id2token


def load_bow_and_sentence_embedding(sentence_bert_model_name: str = None, raw_json_file: str = None,
                                    raw_vocab_file: str = None, normalization: Optional[str] = None, batch_size: int = 64):
    sentence_bert_model = SentenceBertModel(sentence_bert_model_name)
    config = load_config(sentence_bert_model_name)

    dataset_with_bow, id2token = load_bow(
        raw_json_file, raw_vocab_file, normalization)

    # map text_for_contextual to contextual embedding
    def batch_map(batch):
        return {
            'contextual': sentence_bert_model.sentences_to_embeddings(batch['text_for_contextual'])}

    dataset = dataset_with_bow.map(
        batch_map, batched=True, batch_size=batch_size)

    return dataset, config, id2token


def load_config(sentence_bert_model_name: str = None):
    return SentenceBertModel.get_config(sentence_bert_model_name)
