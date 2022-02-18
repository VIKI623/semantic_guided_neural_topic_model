from os.path import basename, join, exists
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from semantic_guided_neural_topic_model.data_modules.utils import load_bow


class BOWDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int = 256, normalization: Optional[str] = None,
                 num_workers: int = 4):
        """
        config data arguments
        :param dataset_dir: the directory containing {dataset_name}.json
        :param batch_size: batch size
        :param normalization: bow representation normalization method: ("average", "tfidf"， None)
        """
        super().__init__()
        self.data_dir = dataset_dir
        self.batch_size = batch_size
        self.dataset_name = basename(dataset_dir)
        self.raw_json_file = join(dataset_dir, self.dataset_name + ".json")
        self.raw_vocab_file = join(dataset_dir, self.dataset_name + ".vocab")
        self.num_workers = num_workers
        if not exists(self.raw_vocab_file):
            self.raw_vocab_file = None

        self.dataset, self.id2token = load_bow(raw_json_file=self.raw_json_file, raw_vocab_file=self.raw_vocab_file,
                                               normalization=normalization)
        self.dataset.set_format(type='torch', columns=['bow'])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=self.num_workers)
