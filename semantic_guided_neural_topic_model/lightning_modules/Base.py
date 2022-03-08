from semantic_guided_neural_topic_model.utils.evaluation import get_external_topic_coherence_batch, get_internal_topic_coherence_batch, get_topics_diversity, BestCoherenceScoreRecorder, AVERAGE_SCORES
from pytorch_lightning import LightningModule
from typing import Mapping, Sequence

class NeuralTopicModelEvaluable(LightningModule):
    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], metric: str='c_npmi', 
                 exclude_external=True):
        super().__init__()
        self.id2token = id2token
        self.reference_texts = reference_texts
        self.exclude_external = exclude_external
        self.best_coherence_score_recorder = BestCoherenceScoreRecorder(exclude_external=exclude_external)
        self.metric = metric
        
    def get_topics(self, top_k: int=10):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self, validation_step_outputs):
        topics = self.get_topics()
        diversity = get_topics_diversity(topics)
        
        internal_topic_coherence = get_internal_topic_coherence_batch(topics, self.reference_texts, processes=4)
        if not self.exclude_external:
            external_topic_coherence = get_external_topic_coherence_batch(topics)
            self.best_coherence_score_recorder.coherence = (topics, diversity, external_topic_coherence, internal_topic_coherence)
        else:
            self.best_coherence_score_recorder.coherence = (topics, diversity, internal_topic_coherence)

        # log score
        if not self.exclude_external:
            self.log('external_topic_coherence', external_topic_coherence[AVERAGE_SCORES])
        self.log('internal_topic_coherence', internal_topic_coherence[AVERAGE_SCORES])
        self.log('diversity', diversity)
        
        # log ckpt metric
        self.log(self.metric, internal_topic_coherence[AVERAGE_SCORES][self.metric])
        
    def get_best_coherence_score(self):
        return self.best_coherence_score_recorder.coherence
        