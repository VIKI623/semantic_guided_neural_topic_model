import argparse
import os
from os.path import join

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from semantic_guided_neural_topic_model.data_modules import BOWDataModule
from semantic_guided_neural_topic_model.experiments.utils import configure_model_checkpoints, dataset_names, topic_nums
from semantic_guided_neural_topic_model.lightning_modules import ProdLDA
from semantic_guided_neural_topic_model.utils import output_dir, data_dir
from semantic_guided_neural_topic_model.utils.persistence import save_json

seed_everything(seed=42)
model_name = "prodlda"

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(
        description="Config arguments to run the model!")
    parser.add_argument('-dn', '--dataset_name', help='dataset name', type=str, choices=('20_news_group',
                                                                                         '5234_event_tweets',
                                                                                         'tag_my_news',
                                                                                         ),
                        default='20_news_group')
    parser.add_argument('-bs', '--batch_size',
                        help='batch size', type=int, default=256)
    parser.add_argument('-norm', '--normalization', help='normalization method',
                        choices=('tfidf', None), default=None)
    parser.add_argument('-tn', '--topic_num',
                        help='topic num', type=int, default=20)
    parser.add_argument('-hd', '--encoder_hidden_dim',
                        help='encoder hidden dim', type=int, default=100)
    parser.add_argument('-do', '--dropout',
                        help='dropout rate', type=float, default=0.2)
    parser.add_argument('-m', '--metric', help='coherence evaluation metric', type=str,
                        default='c_npmi')
    parser.add_argument(
        '-v', '--version', help='version name to log', type=str, default='0221-19')
    parser.add_argument('-en', '--epoch_num',
                        help='epoch num', type=int, default=200)

    args = parser.parse_args()
    normalization = args.normalization
    metric = args.metric
    version = args.version
    batch_size = args.batch_size
    encoder_hidden_dim = args.encoder_hidden_dim
    dropout = args.dropout
    epoch_num = args.epoch_num

    # dataset_name = args.dataset_name
    # topic_num = args.topic_num

    dataset_names = ('5234_event_tweets',)
    topic_nums = (100,)

    for dataset_name in dataset_names:
        for topic_num in topic_nums:

            print("*" * 100)
            print(f"\ndataset: {dataset_name} | topic_num: {topic_num} | model_name: {model_name}\n")
            print("*" * 100)

            # data
            dataset_dir = join(data_dir, dataset_name)
            data_module = BOWDataModule(
                dataset_dir=dataset_dir, batch_size=batch_size, normalization=normalization)
            save_dir = join(output_dir, dataset_name, str(topic_num))

            # model
            model = ProdLDA(id2token=data_module.id2token, reference_texts=data_module.dataset['text_for_bow'],
                            encoder_hidden_dim=encoder_hidden_dim, topic_num=topic_num, dropout=dropout, affine=False,
                            alpha=None, metric=metric)

            # logger
            tb_logger = pl_loggers.TensorBoardLogger(
                save_dir=save_dir, name=model_name, version=version)

            # checkpoint
            callbacks = configure_model_checkpoints(monitor=metric, save_dir=join(save_dir, model_name),
                                                    version=version)

            # trainer
            trainer = Trainer(logger=tb_logger, max_epochs=epoch_num, check_val_every_n_epoch=10, callbacks=callbacks,
                              num_sanity_val_steps=1, log_every_n_steps=40, gpus=1)

            # train
            trainer.fit(model, data_module)

            # save best coherence score
            save_json(join(save_dir, model_name,
                      f'{version}.json'), model.get_best_coherence_score())
