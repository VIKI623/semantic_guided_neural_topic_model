import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from random import random
from typing import Mapping, Sequence, Any

import requests

from semantic_guided_neural_topic_model.utils.log import logger

palmetto_endpoint = r"http://10.208.62.13:7777/service/{}?words={}"

all_coherence_types = ("ca", "cp", "npmi", "cv")


def get_topic_coherence_batch(topics: Sequence[Sequence[str]], coherence_types: Sequence[str] = all_coherence_types) \
        -> Mapping[str, Any]:
    """

    :param topics: list of topic words
    :param coherence_types: list of coherence_type, support "ca", "cp", "npmi", "cv"
    :return: coherence scores less than 1
    """
    # check legal
    for coherence_type in coherence_types:
        assert coherence_type in all_coherence_types

    # build request urls
    request_urls = []
    for coherence_type in coherence_types:
        for topic in topics:
            request_topic = "%20".join(topic)
            request_url = palmetto_endpoint.format(coherence_type, request_topic)
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
                logger.error(str(e))
        return float(r.text)

    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(query_for_topic_coherence, request_urls))

    # build return value
    topics_scores = [{"words": topics[topic_id],
                      "scores": {coherence_type: scores[len(topics) * idx + topic_id]
                                 for idx, coherence_type in enumerate(coherence_types)}
                      } for topic_id in range(len(topics))]
    average_scores = {coherence_type: (
            sum(topic_score["scores"][coherence_type] for topic_score in topics_scores) / len(topics_scores))
        for coherence_type in coherence_types}

    diversity = len(set(word for topic in topics for word in topic)) / sum(len(topic) for topic in topics)
    return {"average_scores": average_scores,
            "items": topics_scores,
            "diversity": diversity
            }


class BestCoherenceScore:
    def __init__(self, metric: str = 'ca'):
        self._coherence = None
        self.score = float('-inf')
        self.metric = metric

    @property
    def coherence(self):
        return self._coherence

    @coherence.setter
    def coherence(self, value):
        new_score = value['average_scores'][self.metric]
        if new_score < self.score:
            return
        self.score = new_score
        self._coherence = value


if __name__ == '__main__':
    start = time.time()
    words = ["cake", "apple", "banana", "cherry", "chocolate"]
    topics = [words] * 100
    pprint(get_topic_coherence_batch(topics))
    print(f"cost {time.time() - start} seconds")
