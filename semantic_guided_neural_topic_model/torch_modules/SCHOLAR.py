import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class torchScholar(nn.Module):

    def __init__(self, vocab_size, embedding_dim, topic_num, n_topic_covars, alpha, bg_init=None, use_interactions=False):
        super().__init__()

        # load the configuration
        self.vocab_size = vocab_size
        self.words_emb_dim = embedding_dim
        self.n_topics = topic_num
        self.n_topic_covars = n_topic_covars
        self.use_interactions = use_interactions
        
        
        # const variable
        update_embeddings = True
        init_emb = None
        self.classifier_layers = 1
        self.n_prior_covars = 0
        self.l1_beta_reg = 0
        self.l1_beta_c_reg = 0
        self.l1_beta_ci_reg = 0
        self.l2_prior_reg = 0
        self.classify_from_covars = False

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.embeddings_x_layer = nn.Linear(self.vocab_size, self.words_emb_dim, bias=False)
        emb_size = self.words_emb_dim
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb))
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics)))
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics)))
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels)
            else:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim)
                self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar)
        self.prior_logvar.requires_grad = False

    def forward(self, X, Y, PC, TC, compute_loss=True, do_average=True, eta_bn_prop=1.0, var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """

        # embed the word counts
        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp()

        # sample noise from a standard normal
        eps = X.data.new().resize_as_(posterior_mean_bn.data).normal_()

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        #eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)

            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1)
            else:
                classifier_input = theta

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        self.prior_logvar = self.prior_logvar.type_as(posterior_logvar)
        self.prior_mean = self.prior_mean.type_as(posterior_mean)
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):

        # compute reconstruction loss
        NL = -(X * (X_recon+1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon+1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD