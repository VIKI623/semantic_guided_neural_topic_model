import argparse
from os.path import join
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from semantic_guided_neural_topic_model.data_modules import TextDataModule
from semantic_guided_neural_topic_model.experiments.utils import configure_model_checkpoints, dataset_names, topic_nums
from semantic_guided_neural_topic_model.lightning_modules import CTM
from semantic_guided_neural_topic_model.utils import output_dir, data_dir
from semantic_guided_neural_topic_model.utils.persistence import save_json, mkdirs
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from semantic_guided_neural_topic_model.utils.evaluation import get_topics_diversity, get_internal_topic_coherence_batch, \
    get_external_topic_coherence_batch, BestCoherenceScoreRecorder


seed_everything(seed=42)
model_name = "ctm_simple"

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(
        description="Config arguments to run the model!")
    parser.add_argument('-dn', '--dataset_name', help='dataset name', type=str, choices=('20_news_group',
                                                                                         '5234_event_tweets',
                                                                                         'tag_my_news',
                                                                                         ),
                        default='5234_event_tweets')
    parser.add_argument('-tn', '--topic_num',
                        help='topic num', type=int, default=20)
    parser.add_argument('-m', '--metric', help='coherence evaluation metric', type=str,
                        default='c_npmi')
    parser.add_argument(
        '-v', '--version', help='version name to log', type=str, default='0224-23')
    parser.add_argument('-en', '--epoch_num',
                        help='epoch num', type=int, default=200)
    parser.add_argument('-sb', '--sentence_bert_model_name', help='sentence bert model name', type=str,
                        default='/data/home/zhaoxin/sentence_transformers/paraphrase-distilroberta-base-v2/0_Transformer')
    parser.add_argument('-wn', '--worker_num',
                        help='worker num', type=int, default=8)

    args = parser.parse_args()
    metric = args.metric
    version = args.version
    sentence_bert_model_name = args.sentence_bert_model_name
    epoch_num = args.epoch_num
    worker_num = args.worker_num

    # dataset_name = args.dataset_name
    # topic_num = args.topic_num

    dataset_names = ('20_news_group',
                     '5234_event_tweets',
                     'tag_my_news',
                     )
    topic_nums = (20, 30, 50)
    # topic_nums = (75, 100)

    for dataset_name in dataset_names:
        for topic_num in topic_nums:

            print("*" * 100)
            print(
                f"\ndataset: {dataset_name} | topic_num: {topic_num} | model_name: {model_name}\n")
            print("*" * 100)

            # mkdirs
            save_dir = join(output_dir, dataset_name,
                            str(topic_num), model_name)
            mkdirs(save_dir)

            # data
            dataset_dir = join(data_dir, dataset_name)
            # sentence_bert_model_name: str, dataset_dir: str, batch_size: int = 256
            data_module = TextDataModule(
                dataset_dir=dataset_dir, num_workers=worker_num)
            qt = TopicModelDataPreparation(sentence_bert_model_name)
            training_dataset = qt.fit(text_for_contextual=data_module.dataset['text_for_contextual'],
                                      text_for_bow=[' '.join(words) for words in data_module.dataset['text_for_bow']])

            # model
            ctm = CombinedTM(bow_size=len(
                qt.vocab), contextual_size=768, n_components=topic_num, num_epochs=epoch_num)

            # train
            ctm.fit(training_dataset)  # run the model

            # induction
            topics = list(ctm.get_topics().values())

            # evaluation
            diversity = get_topics_diversity(topics)
            external_topic_coherence = get_external_topic_coherence_batch(
                topics)
            internal_topic_coherence = get_internal_topic_coherence_batch(
                topics, data_module.dataset['text_for_bow'], processes=worker_num)

            # save best coherence score
            best_coherence_score_recorder = BestCoherenceScoreRecorder()
            best_coherence_score_recorder.coherence = (
                topics, diversity, external_topic_coherence, internal_topic_coherence)
            save_json(join(save_dir, f'{version}.json'),
                      best_coherence_score_recorder.coherence)
