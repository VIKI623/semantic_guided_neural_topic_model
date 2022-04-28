from lib2to3.pgen2 import token
from os.path import basename, join, exists
from typing import Optional
import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from semantic_guided_neural_topic_model.data_modules.utils import load_discrete_covar_mappings, \
    load_bow_and_metadata_for_scholar, BatchSchedulerSampler, load_embedding_for_bert, load_continuous_covar_for_bert
from semantic_guided_neural_topic_model.utils.persistence import read_json
from semantic_guided_neural_topic_model.utils import pre_download_model_dir, data_dir
from transformers import AutoTokenizer
from torch.utils.data.dataset import ConcatDataset


class MetadataDataModuleBase(LightningDataModule):
    def __init__(self, dataset_dir: str):
        """
        config data arguments
        :param dataset_dir: the directory containing {dataset_name}.json
        """
        super().__init__()
        dataset_name = basename(dataset_dir)

        # metadata
        raw_metadata_conf_file = join(
            dataset_dir, f"{dataset_name}_metadata_conf.json")
        metadata_conf = read_json(raw_metadata_conf_file)

        # attribute
        self.continuous_covars = metadata_conf["continuous_covars"]
        self.discrete_covars = metadata_conf["discrete_covars"]
        self.time_covar = metadata_conf.get("time_covar")
        self.text_covar = metadata_conf.get("text_covar")

        # discrete_covars_mapping file path
        discrete_covar_mapping_files = {}
        for discrete_covar in self.discrete_covars:
            discrete_covar_mapping_file = join(
                dataset_dir, f"{dataset_name}.{discrete_covar}")
            discrete_covar_mapping_files[discrete_covar] = discrete_covar_mapping_file

        # attribute
        self.discrete_covar_mappings = load_discrete_covar_mappings(
            discrete_covar_mapping_files=discrete_covar_mapping_files)


class MetadataDataModuleForScholar(MetadataDataModuleBase):
    def __init__(self, dataset_dir: str, batch_size: int = 256, normalization: Optional[str] = None,
                 num_workers: int = 4):
        """
        config data arguments
        :param dataset_dir: the directory containing {dataset_name}.json
        :param batch_size: batch size
        :param normalization: bow representation normalization method: ("average", "tfidf", None)
        """
        super().__init__(dataset_dir=dataset_dir)

        dataset_name = basename(dataset_dir)
        self.covar_size = len(self.continuous_covars) + \
            sum(len(self.discrete_covar_mappings[discrete_covar]["token2id"])
                for discrete_covar in self.discrete_covars) + 1

        self.data_dir = dataset_dir
        self.batch_size = batch_size

        raw_json_file = join(dataset_dir, dataset_name + ".json")
        raw_vocab_file = join(dataset_dir, dataset_name + ".vocab")

        self.num_workers = num_workers
        if not exists(raw_vocab_file):
            raw_vocab_file = None

        self.dataset, self.id2token, self.continuous_covar_scalers = load_bow_and_metadata_for_scholar(raw_json_file=raw_json_file, raw_vocab_file=raw_vocab_file,
                                                                                                       normalization=normalization, batch_size=64,
                                                                                                       continuous_covars=self.continuous_covars, discrete_covars=self.discrete_covars,
                                                                                                       time_covar=self.time_covar, discrete_covar_mappings=self.discrete_covar_mappings)

        self.dataset.set_format(type='torch', columns=['bow', 'covar'])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=self.num_workers)


class MetadataDataModuleForBERT(MetadataDataModuleBase):
    def __init__(self, dataset_dir: str, batch_size: int = 32, num_workers: int = 4,
                 model_name='paraphrase-distilroberta-base-v2', have_text_covar=True):

        super().__init__(dataset_dir=dataset_dir)

        if have_text_covar:
            have_text_covar = self.text_covar is not None
        names = self.continuous_covars + self.discrete_covars

        self.batch_size = batch_size
        self.num_workers = num_workers

        tokenizer = AutoTokenizer.from_pretrained(
            join(pre_download_model_dir, model_name))

        self.bert_encode_keys = tokenizer.model_input_names

        covar2dataset = {}
        self.covar2scaler = {}
        self.discrete_covar2class_num = {}

        for covar in self.continuous_covars:
            dataset = datasets.load_dataset(
                dataset_dir, name=covar, split=datasets.Split.TRAIN)
            dataset = load_embedding_for_bert(
                dataset=dataset, tokenizer=tokenizer, have_text_covar=have_text_covar, have_remove_columns=True)
            covar2dataset[covar], self.covar2scaler[covar] = load_continuous_covar_for_bert(
                dataset=dataset)

        for covar in self.discrete_covars:
            dataset = datasets.load_dataset(
                dataset_dir, name=covar, split=datasets.Split.TRAIN)
            covar2dataset[covar] = load_embedding_for_bert(
                dataset=dataset, tokenizer=tokenizer, have_text_covar=have_text_covar, have_remove_columns=True)
            self.discrete_covar2class_num[covar] = dataset.features['label'].length

        if self.time_covar:
            dataset = datasets.load_dataset(
                dataset_dir, name=self.time_covar, split=datasets.Split.TRAIN)
            dataset = load_embedding_for_bert(
                dataset=dataset, tokenizer=tokenizer, have_text_covar=have_text_covar, have_remove_columns=True)
            covar2dataset[self.time_covar], self.covar2scaler[self.time_covar] = load_continuous_covar_for_bert(
                dataset=dataset)
            names.append(self.time_covar)

        self.task_id_class_label = datasets.ClassLabel(names=names)

        # set format
        for covar, dataset in covar2dataset.items():
            dataset.set_format(type="torch", columns=dataset.column_names)
            print(f"{covar}: {len(dataset)}")
        print(f"have_text_covar: {have_text_covar}")

        self.dataset = ConcatDataset(covar2dataset.values())

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, sampler=BatchSchedulerSampler(dataset=self.dataset,
                                                                              batch_size=self.batch_size, mode='Train'),
                          batch_size=self.batch_size, shuffle=False)

    # def val_dataloader(self):
    #     return DataLoader(dataset=self.dataset, sampler=BatchSchedulerSampler(dataset=self.dataset,
    #                                                                           batch_size=self.batch_size, mode='Valid'),
    #                       batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    dataset_name = "twitter_news_events"
    dataset_dir = join(data_dir, dataset_name)
    dm = MetadataDataModuleForBERT(dataset_dir=dataset_dir)

    print(len(dm.train_dataloader()))
