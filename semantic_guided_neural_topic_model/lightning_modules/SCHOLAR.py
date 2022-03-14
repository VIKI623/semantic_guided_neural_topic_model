import numpy as np
import torch
import torch.nn as nn
from semantic_guided_neural_topic_model.torch_modules.SCHOLAR import torchScholar
from semantic_guided_neural_topic_model.lightning_modules.Base import NeuralTopicModelEvaluable
from typing import Mapping, Sequence, Any, Optional
from itertools import chain
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import get_topics

class Scholar(NeuralTopicModelEvaluable):

    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], encoder_hidden_dim=100,
                 topic_num=50, alpha=1.0, covars_size=None, update_embeddings=True, init_bg=None, update_background=True, 
                 metric='c_npmi'):
        """
        Create the model
        :param alpha: hyperparameter for the document representation prior
        :param update_embeddings: if True, update word embeddings during training
        :param init_bg: a vector of empirical log backgound frequencies
        :param update_background: if True, update the background term during training
        """

        super().__init__(id2token=id2token, reference_texts=reference_texts, metric=metric)

        self.save_hyperparameters()
        self.automatic_optimization = True
        self.topic_num = topic_num
        vocab_size = len(id2token)

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * \
                np.ones((1, self.topic_num)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.topic_num

        # create the pyTorch model
        self.model = torchScholar(vocab_size, encoder_hidden_dim, topic_num=self.topic_num, n_topic_covars=covars_size, 
                                  alpha=self.alpha, bg_init=init_bg, use_interactions=False)

    def configure_optimizers(self):
        grad_params = filter(lambda p: p.requires_grad,
                             self.model.parameters())
        optimizer = torch.optim.Adam(
            grad_params, lr=0.002, betas=(0.99, 0.999))
        return optimizer
    
    def forward(self, batch):
        x_bow = batch['bow']
        covars = batch['covar']
        thetas, _, _, _ = self.model(x_bow, None, None, covars)
        return thetas
    
    def training_step(self, batch, batch_idx):
        
        x_bow = batch['bow']
        covars = batch['covar']
        _, _, _, losses = self.model(x_bow, None, None, covars)
        loss, nl, kld = losses
        
        # log loss
        loss_dict = {'bow_recon_loss': nl, 'kl_loss': kld,
                     'train_loss': loss}
        self.log('train_loss', loss_dict)
        
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        return self(batch)
    
    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            topic_word_dist = self.model.beta_layer.weight.T
            topic_word_dist = torch.softmax(topic_word_dist, dim=-1)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics
