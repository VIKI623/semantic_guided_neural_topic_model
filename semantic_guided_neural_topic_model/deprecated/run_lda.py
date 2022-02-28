from gensim.models.ldamulticore import LdaMulticore
from semantic_guided_neural_topic_model.experiments.utils import topic_nums, dataset_names
from semantic_guided_neural_topic_model.utils import output_dir, data_dir
from os.path import join
from semantic_guided_neural_topic_model.data_modules.bow_datamodule import BOWDataModule
from gensim.corpora.dictionary import Dictionary
from pytorch_lightning import seed_everything
import argparse
from semantic_guided_neural_topic_model.utils.persistence import mkdirs, save_json
from semantic_guided_neural_topic_model.utils import output_dir
from semantic_guided_neural_topic_model.utils.evaluation import get_external_topic_coherence_batch, get_internal_topic_coherence_batch, get_topics_diversity, BestCoherenceScoreRecorder


seed_everything(seed=42)
model_name = "lda"

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(
        description="Config arguments to run the model!")
    parser.add_argument('-dn', '--dataset_name', help='dataset name', type=str, choices=('20_news_group',
                                                                                         '5234_event_tweets',
                                                                                         'tag_my_news',
                                                                                         ),
                        default='20_news_group')
    parser.add_argument('-tn', '--topic_num',
                        help='topic num', type=int, default=20)
    parser.add_argument(
        '-v', '--version', help='version name to log', type=str, default='0228-10')
    parser.add_argument('-en', '--epoch_num',
                        help='epoch num', type=int, default=200)
    parser.add_argument('-wn', '--worker_num',
                        help='worker num', type=int, default=8)

    args = parser.parse_args()
    version = args.version
    epoch_num = args.epoch_num
    worker_num = args.worker_num
    
    dataset_names = ('20_news_group',
                    '5234_event_tweets',
                    'tag_my_news',
                    )
    topic_nums = (20, 30, 50)
    # topic_nums = (75, 100)

    for dataset_name in dataset_names:
        for topic_num in topic_nums:

            print("*" * 100)
            print(f"\ndataset: {dataset_name} | topic_num: {topic_num} | model_name: {model_name}\n")
            print("*" * 100)

            # mkdirs
            save_dir = join(output_dir, dataset_name,
                            str(topic_num), model_name)
            mkdirs(save_dir)

            # data
            dataset_dir = join(data_dir, dataset_name)
            data_module = BOWDataModule(dataset_dir=dataset_dir)
            texts = data_module.dataset['text_for_bow']

            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            # train
            model = LdaMulticore(corpus=corpus, num_topics=topic_num,
                                 id2word=dictionary, iterations=epoch_num, workers=worker_num)

            # save
            model.save(join(save_dir, f"model_{version}_ckpt"))

            # induction
            topic_prob_pairs = model.show_topics(
                num_topics=topic_num, num_words=10, formatted=False)
            topics = [[keyword for keyword, prob in topic_prob_pair[1]]
                      for topic_prob_pair in topic_prob_pairs]

            # evaluation
            diversity = get_topics_diversity(topics)
            external_topic_coherence = get_external_topic_coherence_batch(
                topics)
            internal_topic_coherence = get_internal_topic_coherence_batch(
                topics, texts, processes=worker_num)

            # save best coherence score
            best_coherence_score_recorder = BestCoherenceScoreRecorder()
            best_coherence_score_recorder.coherence = (
                topics, diversity, external_topic_coherence, internal_topic_coherence)
            save_json(join(save_dir, f'{version}.json'),
                      best_coherence_score_recorder.coherence)
