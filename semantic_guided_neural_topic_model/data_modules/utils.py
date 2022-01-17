from collections import Counter
from functools import partial
from typing import Mapping, Iterable, Optional

import torch
from datasets import load_dataset
from gensim.models import TfidfModel

from semantic_guided_neural_topic_model.pretrain_language_models.models import SentenceBertModel
from semantic_guided_neural_topic_model.utils.text_handler import read_raw_vocab, build_bow_vocab


def load_dataset_and_filter(raw_json_file: str, no_below: int = 2):
    dataset = load_dataset('json', data_files=raw_json_file)['train']
    # filter
    dataset = dataset.filter(lambda sample: len(sample['text_for_bow']) >= no_below)
    return dataset


def bow_to_tensor(bow: Iterable[str], token2id: Mapping[str, int], normalization: Optional[str] = None,
                  tfidf_model: Optional[TfidfModel] = None) -> torch.Tensor:
    bow = Counter(token2id[word] for word in bow).most_common()

    if normalization is not None:
        normalization = normalization.lower()
        if normalization == 'tfidf':
            assert tfidf_model is not None
            bow = tfidf_model[bow]
        elif normalization == 'average':
            total_count = sum(count for word, count in bow)
            bow = [(word, count / total_count) for word, count in bow]

    bow_tensor = torch.zeros(len(token2id))
    item = list(zip(*bow))
    bow_tensor[list(item[0])] = torch.tensor(list(item[1])).float()
    return bow_tensor


def load_bow(raw_json_file: str = None, raw_vocab_file: str = None, normalization: Optional[str] = None):
    dataset = load_dataset_and_filter(raw_json_file)
    if raw_json_file:
        id2token, token2id = load_vocab(raw_vocab_file)
    else:
        dictionary = build_bow_vocab(texts=dataset["text_for_bow"])
        id2token, token2id = dict(dictionary.id2token), dict(dictionary.token2id)

    # map text_for_bow to bow
    if normalization and normalization.lower() == 'tfidf':
        tfidf_model = TfidfModel(
            Counter(token2id[word] for word in bow).most_common() for bow in dataset['text_for_bow'])
        bow_to_tensor_with_vocab = partial(bow_to_tensor, token2id=token2id, normalization=normalization,
                                           tfidf_model=tfidf_model)
    else:
        bow_to_tensor_with_vocab = partial(bow_to_tensor, token2id=token2id, normalization=normalization,
                                           tfidf_model=None)

    dataset = dataset.map(lambda item: {"bow": bow_to_tensor_with_vocab(item["text_for_bow"])},
                          batched=False)

    return dataset, id2token


def load_bow_and_sentence_embedding(sentence_bert_model_name: str = None, raw_json_file: str = None,
                                    raw_vocab_file: str = None):
    sentence_bert_model = SentenceBertModel(sentence_bert_model_name)
    config = load_config(sentence_bert_model_name)

    dataset_with_bow, id2token = load_bow(raw_json_file, raw_vocab_file)

    # map text_for_contextual to contextual embedding
    def batch_map(batch):
        return {
            'contextual': sentence_bert_model.sentences_to_embeddings(batch['text_for_contextual'])}

    dataset = dataset_with_bow.map(batch_map, batched=True, batch_size=128)
    dataset.set_format(type='torch', columns=['bow', 'contextual'])

    return dataset, config, id2token


def load_vocab(raw_vocab_file=None):
    id2token, token2id = read_raw_vocab(raw_vocab_file)
    return id2token, token2id


def load_config(sentence_bert_model_name: str = None):
    return SentenceBertModel.get_config(sentence_bert_model_name)
