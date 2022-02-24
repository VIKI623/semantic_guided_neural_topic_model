from contextualized_topic_models.networks.decoding_network import DecoderNetwork
from semantic_guided_neural_topic_model.lightning_modules.Base import NeuralTopicModelEvaluable
from typing import Mapping, Sequence
import torch
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import compute_kl_loss, get_topics
from torch.nn import functional as F

class CTM(NeuralTopicModelEvaluable):
    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], encoder_hidden_dim=100, 
                 topic_num=50, contextual_dim=768, dropout=0.2, metric: str = 'c_npmi'):
        super().__init__(id2token=id2token, reference_texts=reference_texts, metric=metric)
        
        self.save_hyperparameters()
        
        self.automatic_optimization = True
        self.topic_num = topic_num
        vocab_size = len(id2token)
        
        self.model = DecoderNetwork(vocab_size, contextual_dim, 'combined', topic_num, 'prodLDA', 
                                    (encoder_hidden_dim, encoder_hidden_dim), 'softplus', dropout, True, 0)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.model.parameters(), lr=2e-3, betas=(0.99, 0.99))
        return optimizer
    
    def forward(self, x_bow, x_ce):
        posterior_mu, _ = self.inf_net(x_bow, x_ce, None)
        theta = F.softmax(posterior_mu, dim=-1)
        return theta
        
    
    def training_step(self, batch, batch_idx):
        x_bow = batch['bow']
        x_ce = batch['contextual']
        
        
        prior_mean, prior_variance, posterior_mean, posterior_variance,\
        posterior_log_variance, word_dists, _ = self.model(x_bow, x_ce, None)
        
        # calculate loss
        rec_loss = -torch.sum(x_bow * torch.log(word_dists + 1e-10))
        
        # calculate kld
        prior_log_variance = torch.log(prior_variance)
        kl_loss = compute_kl_loss(self.topic_num, prior_mean, posterior_mean, prior_log_variance, posterior_log_variance, 
                                  prior_variance, posterior_variance)
        
        # total loss
        train_loss = rec_loss + kl_loss
        
        # log loss
        loss_dict = {'bow_recon_loss': rec_loss, 'kl_loss': kl_loss,
                     'train_loss': train_loss}
        self.log('train_loss', loss_dict)

        return train_loss
    
    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            topic_word_dist = self.model.beta
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics
    