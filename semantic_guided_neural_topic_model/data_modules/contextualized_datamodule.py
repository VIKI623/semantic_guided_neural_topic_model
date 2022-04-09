from os.path import basename, join, exists
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from semantic_guided_neural_topic_model.data_modules.utils import load_bow_and_sentence_embedding
from semantic_guided_neural_topic_model.utils.persistence import read_json


class ContextualizedDataModule(LightningDataModule):
    def __init__(self, sentence_bert_model_name: str, dataset_dir: str, batch_size: int = 256,
                 normalization: Optional[str] = None, num_workers: int = 4):
        super().__init__()
        self.data_dir = dataset_dir
        self.batch_size = batch_size
        dataset_name = basename(dataset_dir)
        raw_json_file = join(dataset_dir, dataset_name + ".json")
        raw_vocab_file = join(dataset_dir, dataset_name + ".vocab")

        # text covar
        raw_metadata_conf_file = join(
            dataset_dir, f"{dataset_name}_metadata_conf.json")
        text_covar = None
        if exists(raw_metadata_conf_file):
            metadata_conf = read_json(raw_metadata_conf_file)
            text_covar = metadata_conf.get("text_covar")

        self.num_workers = num_workers
        if not exists(raw_vocab_file):
            raw_vocab_file = None

        self.dataset, self.config, self.id2token = load_bow_and_sentence_embedding(
            sentence_bert_model_name=sentence_bert_model_name, raw_json_file=raw_json_file,
            raw_vocab_file=raw_vocab_file, normalization=normalization, text_covar=text_covar)
        self.dataset.set_format(type='torch', columns=['bow', 'contextual'])

        self.contextual_dim = self.config.hidden_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=self.num_workers)
