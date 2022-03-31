from os.path import basename, join, exists
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from semantic_guided_neural_topic_model.data_modules.utils import load_text_and_vocab


class TextDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, num_workers: int = 0):
        """
        config data arguments
        :param dataset_dir: the directory containing {dataset_name}.json
        """
        super().__init__()
        self.data_dir = dataset_dir
        self.dataset_name = basename(dataset_dir)
        raw_json_file = join(dataset_dir, self.dataset_name + ".json")
        raw_vocab_file = join(dataset_dir, self.dataset_name + ".vocab")
        self.num_workers = num_workers
        if not exists(raw_vocab_file):
            raw_vocab_file = None

        self.dataset, self.id2token, _ = load_text_and_vocab(raw_json_file=raw_json_file, raw_vocab_file=raw_vocab_file)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=self.num_workers)
