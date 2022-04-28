from lib2to3.pgen2.token import tok_name
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from functools import partial
from typing import Mapping, Iterable, Optional, Sequence, Tuple
import numpy as np
import datasets
from semantic_guided_neural_topic_model.pretrain_modules.models import SentenceBertModel
from semantic_guided_neural_topic_model.utils.persistence import read_jsons
from semantic_guided_neural_topic_model.utils.text_process import read_raw_vocab, build_bow_vocab
from collections import Counter, defaultdict
from datetime import datetime
from dateutil import parser
from tqdm import tqdm
import math
import torch
from torch.utils.data.sampler import RandomSampler
from sklearnex import patch_sklearn
patch_sklearn()


def filter_dataset_by_attribute_value_length(dataset, attribute_key='text_for_bow', no_below: int = 2):
    # filter
    dataset = dataset.filter(lambda sample: len(
        sample[attribute_key]) >= no_below)
    return dataset


def bow_to_clean_bow(text_for_bow: Iterable[str], token2id: Mapping[str, int]) -> Sequence[str]:
    return [word for word in text_for_bow if word in token2id]


def load_text_and_vocab(raw_json_file: str = None, raw_vocab_file: str = None):
    dataset = datasets.load_dataset('json', data_files=raw_json_file)['train']

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

    return dataset, id2token, token2id


def load_bow(raw_json_file: str = None, raw_vocab_file: str = None, normalization: Optional[str] = None, batch_size: int = 256):
    dataset, id2token, token2id = load_text_and_vocab(
        raw_json_file, raw_vocab_file)

    # map text_for_bow to bow
    handlers = [('bow', CountVectorizer(vocabulary=token2id)), ]
    if normalization and normalization.lower() == 'tfidf':
        handlers.append(('tfidf', TfidfTransformer(norm='l1')))

    pipe = Pipeline(handlers)

    dataset = dataset.map(lambda items: {"bow": pipe.fit_transform(
        [' '.join(item) for item in items["text_for_bow"]]).astype(np.float32).toarray()}, batched=True, batch_size=batch_size)

    return dataset, id2token


def load_bow_and_sentence_embedding(sentence_bert_model_name: str = None, raw_json_file: str = None,
                                    raw_vocab_file: str = None, normalization: Optional[str] = None, batch_size: int = 64,
                                    text_covar: Optional[str] = None):
    sentence_bert_model = SentenceBertModel(sentence_bert_model_name)
    config = load_config(sentence_bert_model_name)
    dataset, id2token = load_bow(
        raw_json_file, raw_vocab_file, normalization, batch_size)

    # map text_for_contextual to contextual embedding
    if text_covar:
        dataset = dataset.flatten()

        def batch_map(batch):
            return {
                'contextual': sentence_bert_model.sentences_to_embeddings(batch['text_for_contextual'], batch[f'metadata.{text_covar}'])}
    else:
        def batch_map(batch):
            return {
                'contextual': sentence_bert_model.sentences_to_embeddings(batch['text_for_contextual'])}

    dataset = dataset.map(
        batch_map, batched=True, batch_size=batch_size, remove_columns=['text_for_contextual'])

    return dataset, config, id2token


def load_bow_and_metadata_for_scholar(raw_json_file: str = None, raw_vocab_file: str = None, normalization: Optional[str] = None,
                                      batch_size: int = 64, continuous_covars: Sequence[str] = None, discrete_covars: Sequence[str] = None,
                                      time_covar: str = None, discrete_covar_mappings: Mapping[str, Mapping[str, Mapping]] = None):

    dataset_with_bow, id2token = load_bow(
        raw_json_file, raw_vocab_file, normalization, batch_size)

    # continuous_covars
    continuous_covar_scalers = {}
    for continuous_covar in continuous_covars:
        X = np.fromiter(iter=(sample["metadata"][continuous_covar]
                        for sample in dataset_with_bow), dtype=np.float32)
        X = np.expand_dims(X, axis=-1)
        scaler = get_standard_scaler(X=X)
        continuous_covar_scalers[continuous_covar] = scaler

    # time_covar
    X = np.fromiter(iter=tqdm((str_to_timestamp(sample["metadata"][time_covar]) for sample in dataset_with_bow),
                              total=len(dataset_with_bow)), dtype=np.float32)
    X = np.expand_dims(X, axis=-1)
    scaler = get_standard_scaler(X=X)
    continuous_covar_scalers[time_covar] = scaler
    del X

    # map metadata to covar embedding
    def batch_map(batch):
        batch_metadata = batch["metadata"]
        batch_size = len(batch_metadata)

        covars = []
        # continuous_covar_scalers
        for continuous_covar in continuous_covars:
            batch_continuous_val = np.fromiter(
                iter=(sample[continuous_covar] for sample in batch_metadata), dtype=np.float32)
            batch_continuous_val = np.expand_dims(
                batch_continuous_val, axis=-1)
            standard_vals = continuous_covar_scalers[continuous_covar].transform(
                batch_continuous_val)  # (batch_size, )
            covars.append(standard_vals)  # (batch_size, 1)

        # time_covar_scalers
        batch_continuous_val = np.fromiter(iter=(str_to_timestamp(sample[time_covar])
                                                 for sample in batch_metadata), dtype=np.float32)
        batch_continuous_val = np.expand_dims(batch_continuous_val, axis=-1)
        standard_vals = continuous_covar_scalers[time_covar].transform(
            batch_continuous_val)
        covars.append(standard_vals)
        del batch_continuous_val, standard_vals

        # discrete_covar_mappings
        for discrete_covar in discrete_covars:
            covar_token_to_id = discrete_covar_mappings[discrete_covar]['token2id']
            class_num = len(covar_token_to_id)
            discrete_covar_matrix = np.zeros(
                (batch_size, class_num), dtype=int)  # (batch_size, class_num)

            for idx, sample in enumerate(batch_metadata):
                covar_vals = (
                    covar_token_to_id[val] for val in sample[discrete_covar] if val in covar_token_to_id)
                val_to_count = Counter(covar_vals)
                for covar_val, count in val_to_count.items():
                    discrete_covar_matrix[idx, covar_val] = count

            covars.append(discrete_covar_matrix)

        return {
            'covar': np.concatenate(covars, axis=-1)}

    dataset = dataset_with_bow.map(
        batch_map, batched=True, batch_size=batch_size)

    return dataset, id2token, continuous_covar_scalers


def load_config(sentence_bert_model_name: str = None):
    return SentenceBertModel.get_config(sentence_bert_model_name)


def str_to_timestamp(datetime_str: str) -> float:
    if isinstance(datetime_str, datetime):
        return datetime_str.timestamp()
    return parser.parse(datetime_str).timestamp()


def timestamp_to_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp)


def get_standard_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X=X)

    return scaler


def load_discrete_covar_mappings(discrete_covar_mapping_files: Mapping[str, str] = None) \
        -> Mapping[str, Mapping[str, Mapping]]:
    discrete_covar_mappings = defaultdict(dict)

    for discrete_covar, discrete_covar_mapping_file in discrete_covar_mapping_files.items():
        id2token, token2id = read_raw_vocab(
            full_path=discrete_covar_mapping_file)
        discrete_covar_mappings[discrete_covar]['id2token'] = id2token
        discrete_covar_mappings[discrete_covar]['token2id'] = token2id

    return discrete_covar_mappings


def load_embedding_for_bert(dataset: datasets.arrow_dataset.Dataset, tokenizer=None, have_text_covar: bool = True, have_remove_columns: bool = True, batch_size: int = 64) \
        -> datasets.arrow_dataset.Dataset:
    def batch_map(batch):
        text = batch["text_for_contextual"]
        text_pair = None
        if have_text_covar:
            text_pair = batch['text_covar']

        encoded_input = tokenizer(text, text_pair, padding='max_length',
                                  truncation='only_second', max_length=tokenizer.model_max_length)

        return encoded_input

    remove_columns = None
    if have_remove_columns:
        remove_columns = ["text_for_contextual", "text_covar"]

    dataset = dataset.map(batch_map, batched=True,
                          batch_size=batch_size, remove_columns=remove_columns)

    return dataset


def load_continuous_covar_for_bert(dataset: datasets.arrow_dataset.Dataset, batch_size: int = 64) \
        -> Tuple[datasets.arrow_dataset.Dataset, StandardScaler]:
    X = np.expand_dims(dataset["label"], axis=-1)
    scaler = get_standard_scaler(X=X)

    def batch_map(batch):
        batch_continuous_covar = np.expand_dims(batch["label"], axis=-1)

        batch_continuous_covar = scaler.transform(batch_continuous_covar)

        return {"label": np.squeeze(batch_continuous_covar, axis=-1)}

    dataset = dataset.map(batch_map, batched=True, batch_size=batch_size)

    return dataset, scaler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch

    source_url: https://blog.csdn.net/sgzqc/article/details/118606389?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3
    """

    def __init__(self, dataset, batch_size, mode='Train'):
        self.mode = mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        if self.mode == 'Train':
            self.largest_dataset_size = max(
                [len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        if self.mode == 'Train':
            return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * self.number_of_datasets
        else:
            return sum([len(cur_dataset) for cur_dataset in self.dataset.datasets])

    def __iter__(self):
        final_samples_list = []  # this is a list of indexes from the combined dataset

        if self.mode == 'Train':
            samplers_list = []
            sampler_iterators = []
            for dataset_idx in range(self.number_of_datasets):
                cur_dataset = self.dataset.datasets[dataset_idx]
                sampler = RandomSampler(cur_dataset)
                samplers_list.append(sampler)
                cur_sampler_iterator = sampler.__iter__()
                sampler_iterators.append(cur_sampler_iterator)

            step = self.batch_size * self.number_of_datasets
            samples_to_grab = self.batch_size
            # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
            epoch_samples = self.largest_dataset_size * self.number_of_datasets
            push_index_val = [0] + self.dataset.cumulative_sizes[:-1]

            for _ in range(0, epoch_samples, step):
                for i in range(self.number_of_datasets):
                    cur_batch_sampler = sampler_iterators[i]
                    cur_samples = []
                    for _ in range(samples_to_grab):
                        try:
                            cur_sample_org = cur_batch_sampler.__next__()
                            cur_sample = cur_sample_org + push_index_val[i]
                            cur_samples.append(cur_sample)
                        except StopIteration:
                            # got to the end of iterator - restart the iterator and continue to get samples
                            # until reaching "epoch_samples"
                            sampler_iterators[i] = samplers_list[i].__iter__()
                            cur_batch_sampler = sampler_iterators[i]
                            cur_sample_org = cur_batch_sampler.__next__()
                            cur_sample = cur_sample_org + push_index_val[i]
                            cur_samples.append(cur_sample)
                    final_samples_list.extend(cur_samples)
        else:
            dataset_size_index = [0] + self.dataset.cumulative_sizes
            for dataset_idx in range(len(dataset_size_index) - 1):
                dataset_start_index = list(range(
                    dataset_size_index[dataset_idx], dataset_size_index[dataset_idx + 1], self.batch_size)) + [dataset_size_index[dataset_idx + 1]]
                for start in range(len(dataset_start_index) - 1):
                    batch = list(
                        range(dataset_start_index[start], dataset_start_index[start + 1]))
                    if (start + 1 == len(dataset_start_index) - 1) and len(batch) != self.batch_size:
                        continue
                    final_samples_list.extend(batch)
        return iter(final_samples_list)
