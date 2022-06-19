from typing import Any, Optional, Callable, Union, Mapping, Sequence
import torch
import torch.nn.functional as F
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import alpha_and_topic_num_to_prior, compute_kl_loss, \
    get_topics, cross_entropy
from semantic_guided_neural_topic_model.torch_modules.VAE import SECT as SECT
from semantic_guided_neural_topic_model.lightning_modules.Base import NeuralTopicModelEvaluable


class SECT(NeuralTopicModelEvaluable):
    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], encoder_hidden_dim=100,
                 topic_num=50, contextual_dim=768, dropout=0.2, ce_loss="MSE", ce_loss_coef=1.0, affine=False, alpha=None,
                 metric='c_npmi'):

        super().__init__(id2token=id2token, reference_texts=reference_texts,
                         metric=metric, exclude_external=True)

        self.save_hyperparameters()

        self.automatic_optimization = True
        self.topic_num = topic_num
        vocab_size = len(id2token)

        if alpha is None:
            alpha = torch.ones(1, topic_num)
        assert alpha.shape == (1, topic_num)

        self.prior_mu, self.prior_logvar, self.prior_var = alpha_and_topic_num_to_prior(
            topic_num, alpha)

        # torch module
        self.model = SECT(encode_dims=(vocab_size, encoder_hidden_dim, encoder_hidden_dim, topic_num),
                          contextual_decode_dims=(topic_num, contextual_dim), bow_decode_dims=(topic_num, vocab_size),
                          contextual_dim=contextual_dim, dropout=dropout, affine=affine)

        print(self.model)

        # ce loss func
        if ce_loss == "INFONCE":
            self.ce_loss_func = self.infonce_loss
        elif ce_loss == "COS":
            self.ce_loss_func = self.cos_loss
        elif ce_loss == "MSE":
            self.ce_loss_func = self.mse_loss

        # ce loss coef
        self.ce_loss_coef = ce_loss_coef

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.002, betas=(0.99, 0.999))
        return optimizer

    def forward(self, x_bow, x_ce):
        return self.model.encode(x_bow, x_ce)

    def on_train_start(self) -> None:
        self.prior_mu = self.prior_mu.to(self.device)
        self.prior_logvar = self.prior_logvar.to(self.device)
        self.prior_var = self.prior_var.to(self.device)

    def training_step(self, batch, batch_idx):
        x_bow = batch['bow']
        x_ce = batch['contextual']

        bow_recon, ce_recon, mu, log_var = self.model(x_bow, x_ce)

        # calculate loss
        bow_rec_loss = cross_entropy(bow_recon, x_bow)
        ce_rec_loss = self.ce_loss_coef * self.ce_loss_func(ce_recon, x_ce)

        # calculate kld
        var = torch.exp(log_var)
        kl_loss = compute_kl_loss(
            self.topic_num, self.prior_mu, mu, self.prior_logvar, log_var, self.prior_var, var)

        # total loss
        train_loss = bow_rec_loss + ce_rec_loss + kl_loss

        # log loss
        loss_dict = {'bow_recon_loss': bow_rec_loss, 'ce_recon_loss': ce_rec_loss, 'kl_loss': kl_loss,
                     'train_loss': train_loss}
        self.log('train_loss', loss_dict)

        return train_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        return self(batch['bow'], batch['contextual'])

    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num, device=self.device)
            topic_word_dist = self.model.decode(idxes)
            topic_word_dist = torch.softmax(topic_word_dist, dim=-1)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics

    def get_topic_contextual_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num).to(self.device)
            topic_contextual_embeddings = self.model.decode_to_ce(idxes)
        return topic_contextual_embeddings

    def infonce_loss(self, ce_recon, ce, temperature=0.05):
        # ce_recon, ce: (batch_size, contextual_dim)
        batch_size = ce_recon.shape[0]
        y_true = torch.arange(batch_size).to(self.device)

        ce_recon = F.normalize(ce_recon, dim=1, p=2)
        ce = F.normalize(ce, dim=1, p=2)
        sim_score = torch.matmul(ce_recon, ce.transpose(0, 1))
        sim_score = sim_score / temperature

        loss = F.cross_entropy(sim_score, y_true, reduction='sum')
        return loss

    def cos_loss(self, ce_recon, ce):
        return F.cosine_embedding_loss(ce_recon, ce, target=torch.ones(len(ce)).to(self.device), reduction='sum')

    @staticmethod
    def mse_loss(ce_recon, ce):
        return F.mse_loss(ce_recon, ce, reduction='sum')
