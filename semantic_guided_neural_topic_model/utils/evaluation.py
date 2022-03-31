import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from random import random
from typing import Mapping, Sequence, Any
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
import requests
from gensim.models import CoherenceModel
from semantic_guided_neural_topic_model.utils.log import logger
from random import choice
from munkres import Munkres

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import k_means


palmetto_endpoints = (
    "http://127.0.0.1:7777/service/{}?words={}",
    "http://127.0.0.1:7778/service/{}?words={}",
)

all_external_coherence_types = ("npmi", "cv")
all_internal_coherence_types = ("c_v", "c_npmi")

# const_key
AVERAGE_SCORES = "average_scores"
EXTERNAL = "external"
INTERNAL = "internal"
TOPICS = "topics"
KEYWORDS = "keywords"
SCORES = "scores"
DIVERSITY = "diversity"


def get_external_topic_coherence_batch(topics: Sequence[Sequence[str]], coherence_types: Sequence[str] = all_external_coherence_types,
                                       processes: int = 4) \
        -> Mapping[str, Any]:
    """

    :param topics: list of topic words
    :param coherence_types: list of coherence_type, support "ca", "cp", "npmi", "cv"
    :return: coherence scores less than 1
    """
    palmetto_endpoint = choice(palmetto_endpoints)

    # check legal
    for coherence_type in coherence_types:
        assert coherence_type in all_external_coherence_types

    # build request urls
    request_urls = []
    for coherence_type in coherence_types:
        for topic in topics:
            request_topic = "%20".join(topic)
            request_url = palmetto_endpoint.format(
                coherence_type, request_topic)
            request_urls.append(request_url)

    # build urls
    def query_for_topic_coherence(request_url):
        while True:
            try:
                r = requests.get(request_url)
                if r.ok:
                    break
            except BaseException as e:
                time.sleep(random() * 4 + 1.0)
                logger.error(f"{palmetto_endpoint} down: {str(e)}")
        return float(r.text)

    with ThreadPoolExecutor(max_workers=processes) as executor:
        scores = list(tqdm(executor.map(query_for_topic_coherence, request_urls), total=len(
            request_urls), desc="external topic coherence under evaluation:"))

    # build return value
    topics_scores = [{coherence_type: scores[len(topics) * idx + topic_id] for idx, coherence_type in enumerate(coherence_types)}
                     for topic_id in range(len(topics))]
    average_scores = {coherence_type: (sum(topic_score[coherence_type] for topic_score in topics_scores) / len(topics_scores))
                      for coherence_type in coherence_types}

    return {AVERAGE_SCORES: average_scores,
            SCORES: topics_scores,
            }


def get_internal_topic_coherence_batch(topics: Sequence[Sequence[str]], reference_texts: Sequence[Sequence[str]],
                                       coherence_types: Sequence[str] = all_internal_coherence_types, processes: int = 1) \
        -> Mapping[str, Any]:

    dictionary = Dictionary(reference_texts)
    average_scores = {}

    for coherence_type in tqdm(coherence_types, desc="internal topic coherence under evaluation:"):
        cm = CoherenceModel(topics=topics, texts=reference_texts,
                            dictionary=dictionary, coherence=coherence_type, processes=processes)
        coherence = cm.get_coherence().item()  # get coherence value
        average_scores[coherence_type] = coherence

    return {AVERAGE_SCORES: average_scores,
            }


def get_topics_diversity(topics: Sequence[Sequence[str]]):
    diversity = len(set(word for topic in topics for word in topic)
                    ) / sum(len(topic) for topic in topics)
    return diversity


class BestCoherenceScoreRecorder:
    def __init__(self, exclude_external=False):
        self.external_topic_coherence = None
        self.internal_topic_coherence = None
        self.diversity = .0
        self.topics = None
        self.exclude_external = exclude_external

    @property
    def coherence(self):
        if self.exclude_external:
            return {
                DIVERSITY: self.diversity,
                AVERAGE_SCORES: {
                    INTERNAL: self.internal_topic_coherence[AVERAGE_SCORES],
                },
                TOPICS: [
                    {
                        KEYWORDS: topic,
                    } for topic
                    in self.topics
                ]
            }

        return {
            DIVERSITY: self.diversity,
            AVERAGE_SCORES: {
                EXTERNAL: self.external_topic_coherence[AVERAGE_SCORES],
                INTERNAL: self.internal_topic_coherence[AVERAGE_SCORES],
            },
            TOPICS: [
                {
                    KEYWORDS: topic,
                    SCORES: {
                        EXTERNAL: external_topic_coherence,
                    },
                } for topic, external_topic_coherence
                in zip(self.topics, self.external_topic_coherence[SCORES])
            ]
        }

    @coherence.setter
    def coherence(self, value):
        if self.exclude_external:
            topics, diversity, internal_topic_coherence = value
        else:
            topics, diversity, external_topic_coherence, internal_topic_coherence = value

        flag = False
        # diversity
        if diversity > self.diversity:
            self.diversity = diversity
            flag = True

        # internal
        if self.internal_topic_coherence is None:
            self.internal_topic_coherence = internal_topic_coherence
            flag = True
        else:
            for coherence_type, score in self.internal_topic_coherence[AVERAGE_SCORES].items():
                candidate_score = internal_topic_coherence[AVERAGE_SCORES].get(
                    coherence_type)
                if candidate_score is not None and candidate_score > score:
                    self.internal_topic_coherence[AVERAGE_SCORES][coherence_type] = candidate_score
                    flag = True

        if not self.exclude_external:
            # external
            if self.external_topic_coherence is None:
                self.external_topic_coherence = external_topic_coherence
                flag = True
            else:
                for coherence_type, score in self.external_topic_coherence[AVERAGE_SCORES].items():
                    candidate_score = external_topic_coherence[AVERAGE_SCORES].get(
                        coherence_type)
                    if candidate_score is not None and candidate_score > score:
                        self.external_topic_coherence[AVERAGE_SCORES][coherence_type] = candidate_score
                        flag = True

        # update topic if better
        if flag is True:
            self.topics = topics
            if not self.exclude_external:
                self.external_topic_coherence[SCORES] = external_topic_coherence[SCORES]


def get_text_clustering_acc(docs_vec: np.array, docs_label: np.array) -> float:
    unique_labels = list(set(docs_label))
    lable_num = len(unique_labels)

    _, docs_cluster_id, *_ = k_means(X=docs_vec, n_clustersint=lable_num)

    matrix = [[0 for _ in range(lable_num)] for _ in range(lable_num)]

    for cluster_id in range(lable_num):
        index = np.array(np.where(docs_cluster_id == cluster_id))
        for label_id in range(lable_num):
            label = unique_labels[label_id]
            label_count = sum(docs_label[index] == label)
            matrix[cluster_id][label_id] = -label_count
    
    indexes = Munkres().compute(matrix)
    
    right_assign_count = -sum(matrix[row][col] for row, col in indexes)
    acc = len(docs_label) / right_assign_count

    return acc
