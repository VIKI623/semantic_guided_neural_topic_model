import argparse
import os
from os.path import join
from os import makedirs
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from datasets import load_dataset
from pytorch_lightning import seed_everything

from semantic_guided_neural_topic_model.experiments.utils import dataset_names, topic_nums
from semantic_guided_neural_topic_model.utils import output_dir, data_dir
from semantic_guided_neural_topic_model.utils.evaluation import get_external_topic_coherence_batch
from semantic_guided_neural_topic_model.utils.persistence import save_json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed_everything(seed=42)
model_name = "ctm"

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Config arguments to run the model!")
    parser.add_argument('-dn', '--dataset_name', help='dataset name', type=str, choices=('20_news_group',
                                                                                         '5234_event_tweets',
                                                                                         'tag_my_news',
                                                                                         ),
                        default='20_news_group')
    parser.add_argument('-bs', '--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('-tn', '--topic_num', help='topic num', type=int, default=20)
    parser.add_argument('-hd', '--encoder_hidden_dim', help='encoder hidden dim', type=int, default=100)
    parser.add_argument('-do', '--dropout', help='dropout rate', type=float, default=0.2)
    parser.add_argument('-v', '--version', help='version name to log', type=str, default='0118-1039')
    parser.add_argument('-en', '--epoch_num', help='epoch num', type=int, default=200)
    parser.add_argument('-sb', '--sentence_bert_model_name', help='sentence bert model name', type=str,
                        default='/data/home/zhaoxin/sentence_transformers/paraphrase-distilroberta-base-v2')

    args = parser.parse_args()
    version = args.version
    batch_size = args.batch_size
    encoder_hidden_dim = args.encoder_hidden_dim
    sentence_bert_model_name = args.sentence_bert_model_name
    dropout = args.dropout
    epoch_num = args.epoch_num

    dataset_name = args.dataset_name
    topic_num = args.topic_num
    dataset_names = (
        '20_news_group',)
    topic_nums = (100, )

    for dataset_name in dataset_names:
        for topic_num in topic_nums:
            print(f"\ndataset:{dataset_name} | topic_num:{topic_num}")

            save_dir = join(output_dir, dataset_name, str(topic_num), model_name)
            makedirs(save_dir, exist_ok=True)

            # data
            qt = TopicModelDataPreparation(sentence_bert_model_name)

            raw_json_file = join(data_dir, dataset_name, f'{dataset_name}.json')
            dataset = load_dataset('json', data_files=raw_json_file)['train']
            dataset = dataset.map(lambda item: {'text_for_bow_str': " ".join(item['text_for_bow'])}, batched=False)

            training_dataset = qt.fit(text_for_contextual=dataset['text_for_contextual'],
                                      text_for_bow=dataset['text_for_bow_str'])

            # model
            ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=topic_num,
                             hidden_sizes=(encoder_hidden_dim, encoder_hidden_dim), batch_size=batch_size, num_epochs=epoch_num,
                             num_data_loader_workers=4, dropout=dropout)

            # train
            ctm.fit(training_dataset, save_dir=save_dir)  # run the model

            topics = list(ctm.get_topics().values())

            save_json(join(save_dir, f'{version}.json'), get_external_topic_coherence_batch(topics))
