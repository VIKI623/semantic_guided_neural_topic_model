import numpy as np
import torch
import torch.nn as nn
from semantic_guided_neural_topic_model.torch_modules.SCHOLAR import torchScholar, simpleScholar
from semantic_guided_neural_topic_model.lightning_modules.Base import NeuralTopicModelEvaluable
from typing import Mapping, Sequence, Any, Optional
from itertools import chain
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import get_topics, alpha_and_topic_num_to_prior, cross_entropy, compute_kl_loss

class SCHOLAR(NeuralTopicModelEvaluable):

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
    
    def predict_step(self, batch: Any, batch_idx: int):
        return self(batch)
    
    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            topic_word_dist = self.model.beta_layer.weight.T
            topic_word_dist = torch.softmax(topic_word_dist, dim=-1)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics


class SimpleSCHOLAR(NeuralTopicModelEvaluable):
    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], encoder_hidden_dim=100, topic_num=50, 
                 dropout=0.2, covars_size=1940, affine=False, alpha=None, metric='c_npmi'):
        super().__init__(id2token=id2token, reference_texts=reference_texts, metric=metric)
        
        self.save_hyperparameters()

        self.automatic_optimization = True
        self.topic_num = topic_num
        vocab_size = len(id2token)

        if alpha is None:
            alpha = torch.ones(1, topic_num)
        assert alpha.shape == (1, topic_num)

        self.prior_mu, self.prior_logvar, self.prior_var = alpha_and_topic_num_to_prior(topic_num, alpha)

        # torch module
        self.model = simpleScholar(vocab_size=vocab_size, hidden_dim=encoder_hidden_dim, topic_num=topic_num, 
                                   covars_size=covars_size, dropout=dropout, affine=affine)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, betas=(0.99, 0.999))
        return optimizer

    def forward(self, x_bow):
        return self.model.encode(x_bow)

    def on_train_start(self) -> None:
        self.prior_mu = self.prior_mu.to(self.device)
        self.prior_logvar = self.prior_logvar.to(self.device)
        self.prior_var = self.prior_var.to(self.device)

    def training_step(self, batch, batch_idx):
        x_bow = batch['bow']
        x_covars = batch['covar']

        x_bow_recon, mu, log_var = self.model(x_bow, x_covars)

        # calculate loss
        rec_loss = cross_entropy(x_bow_recon, x_bow)

        # calculate kld
        var = torch.exp(log_var)
        kl_loss = compute_kl_loss(self.topic_num, self.prior_mu, mu, self.prior_logvar, log_var, self.prior_var, var)

        # total loss
        train_loss = rec_loss + kl_loss

        # log loss
        loss_dict = {'bow_recon_loss': rec_loss, 'kl_loss': kl_loss,
                     'train_loss': train_loss}
        self.log('train_loss', loss_dict)

        return train_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        return self(batch['bow'])

    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num, device=self.device)
            topic_word_dist = self.model.decode(idxes)
            topic_word_dist = torch.softmax(topic_word_dist, dim=-1)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics
