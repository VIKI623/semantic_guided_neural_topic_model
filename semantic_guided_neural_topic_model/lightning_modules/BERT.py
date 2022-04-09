from collections import defaultdict
from email.policy import default
from typing import Sequence, Mapping
from pytorch_lightning import LightningModule
from semantic_guided_neural_topic_model.torch_modules.BERT import BertForMultiTask
from transformers import AdamW, get_linear_schedule_with_warmup
import datasets
import torch
import torch.nn.functional as F
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import get_topics

class SparseMultiMetadataSBERT(LightningModule):
    def __init__(self, model_name: str = 'paraphrase-distilroberta-base-v2', continuous_covars: Sequence[str] = None,
                 discrete_covar2class_num: Mapping[str, int] = None, warmup_proportion: float = 0.1, epoch_steps: int = 10000,
                 task_epoch: int = 20, union_epoch: int = 3, final_task_epoch: int = 10, task_id_class_label: datasets.ClassLabel = None,
                 bert_encode_keys: Sequence[str] = None, adv_train_num_per_step: int = 5, adv_weight_clamp: float = 0.01,
                 save_pretrain_dir: str = None):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.warmup_proportion = warmup_proportion
        self.epoch_steps = epoch_steps
        self.task_id_class_label = task_id_class_label
        self.task_num = self.task_id_class_label.num_classes
        self.adv_train_num_per_step = adv_train_num_per_step
        self.adv_weight_clamp = adv_weight_clamp
        self.save_pretrain_dir = save_pretrain_dir

        # covar category
        self.continuous_covars = continuous_covars
        self.discrete_covar2class_num = discrete_covar2class_num

        assert self.task_num == len(
            continuous_covars) + len(discrete_covar2class_num)

        self.bert_encode_keys = bert_encode_keys

        # train epoch cut
        self.task_epoch = task_epoch
        self.union_epoch = union_epoch
        self.final_task_epoch = final_task_epoch

        self.model = BertForMultiTask(
            model_name=model_name, continuous_covars=continuous_covars, discrete_covar2class_num=discrete_covar2class_num)

    def forward(self, sent_embed):
        covar2dist = {}
        for covar_name, task_head in self.model.covars_head.items():
            covar2dist[covar_name] = task_head(sent_embed)
        return covar2dist

    def configure_optimizers(self):
        # p1: task optimizer
        task_total_steps = self.task_epoch * self.epoch_steps / self.task_num
        task_optimizer = AdamW(
            [p for n, p in self.model.named_parameters() if 'bert' not in n], lr=5e-4)
        task_scheduler = get_linear_schedule_with_warmup(
            task_optimizer,
            num_warmup_steps=self.warmup_proportion * task_total_steps,
            num_training_steps=task_total_steps,
        )

        # p2: union optimizer & scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and ('task_adversarial_head' not in n)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and ('task_adversarial_head' not in n)],
                "weight_decay": 0.0,
            },
        ]
        union_optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

        union_total_steps = self.union_epoch * self.epoch_steps / self.task_num
        union_scheduler = get_linear_schedule_with_warmup(
            union_optimizer,
            num_warmup_steps=self.warmup_proportion * union_total_steps,
            num_training_steps=union_total_steps,
        )

        union_adv_total_steps = self.union_epoch * self.epoch_steps / self.task_num

        union_adv_optimizer = AdamW([p for n, p in self.model.named_parameters(
        ) if 'task_adversarial_head' in n], lr=2e-5)
        union_adv_scheduler = get_linear_schedule_with_warmup(
            union_adv_optimizer,
            num_warmup_steps=self.warmup_proportion * union_adv_total_steps,
            num_training_steps=union_adv_total_steps,
        )

        # p3: fine-tune single task head
        final_task_total_steps = self.final_task_epoch * self.epoch_steps / self.task_num
        final_task_optimizer = AdamW(
            [p for n, p in self.model.named_parameters() if 'bert' not in n], lr=5e-5)
        final_task_scheduler = get_linear_schedule_with_warmup(
            final_task_optimizer,
            num_warmup_steps=self.warmup_proportion * final_task_total_steps,
            num_training_steps=final_task_total_steps,
        )

        return [task_optimizer, union_optimizer, union_adv_optimizer, final_task_optimizer], [task_scheduler, union_scheduler, union_adv_scheduler, final_task_scheduler]

    def on_train_epoch_start(self):
        # set optimizer & scheduler
        task_optimizer, union_optimizer, union_adv_optimizer, final_task_optimizer = self.optimizers()
        task_scheduler, union_scheduler, union_adv_scheduler, final_task_scheduler = self.lr_schedulers()

        if self.trainer.current_epoch < self.task_epoch:
            self.model.freeze_bert()
            self.main_optimizer = task_optimizer
            self.main_scheduler = task_scheduler
            self.adv_optimizer = None
            self.adv_scheduler = None

        elif self.trainer.current_epoch < self.task_epoch + self.union_epoch:
            self.model.unfreeze_bert()
            self.main_optimizer = union_optimizer
            self.main_scheduler = union_scheduler
            self.adv_optimizer = union_adv_optimizer
            self.adv_scheduler = union_adv_scheduler

        else:
            self.model.freeze_bert()
            self.main_optimizer = final_task_optimizer
            self.main_scheduler = final_task_scheduler
            self.adv_optimizer = None
            self.adv_scheduler = None

    def training_step(self, batch, batch_idx):
        # parse batch data
        task_id = batch['task_id'][0].item()
        covar_name = self.task_id_class_label.int2str(task_id)
        encoded_input = {key: batch[key] for key in self.bert_encode_keys}

        # train epoch switch
        if self.task_epoch <= self.trainer.current_epoch < self.task_epoch + self.union_epoch:
            # adv train
            mean_pooling_hidden_states = self.model.bert(**encoded_input)
            mean_pooling_hidden_states = mean_pooling_hidden_states.detach()
            for _ in range(self.adv_train_num_per_step):
                task_prediction = self.model.task_adversarial_head(
                    mean_pooling_hidden_states)
                task_adv_loss_for_head = self._compute_task_adversarial_loss(
                    task_prediction, batch['task_id'])
                self.manual_backward(task_adv_loss_for_head)
                self.adv_optimizer.step()
                self.adv_optimizer.zero_grad()
                # weight clipping
                for para in self.model.task_adversarial_head.parameters():
                    with torch.no_grad():
                        para.clamp_(-self.adv_weight_clamp,
                                    self.adv_weight_clamp)

            # union train
            # forward
            covar_prediction, task_prediction = self.model(
                covar_name=covar_name, **encoded_input)

            # loss
            covar_loss_func = self._choose_loss_func(covar_name=covar_name)
            covar_pred_loss = covar_loss_func(covar_prediction, batch['label'])
            task_adv_loss = self._compute_task_adversarial_loss(
                task_prediction, batch['task_id'])
            loss = covar_pred_loss - task_adv_loss
            self.manual_backward(loss)
        else:
            # forward
            covar_prediction, task_prediction = self.model(
                covar_name=covar_name, **encoded_input)

            # loss
            covar_loss_func = self._choose_loss_func(covar_name=covar_name)
            covar_pred_loss = covar_loss_func(covar_prediction, batch['label'])
            task_adv_loss = self._compute_task_adversarial_loss(
                task_prediction, batch['task_id'])
            loss = covar_pred_loss + task_adv_loss
            self.manual_backward(loss)

        # gradient_accumulation
        if (self.trainer.global_step + 1) % self.task_num == 0:
            self.main_optimizer.step()
            self.main_optimizer.zero_grad()
            self.main_scheduler.step()
            if self.adv_scheduler:
                self.adv_scheduler.step()

        self.log_dict({f"{covar_name}_loss": covar_pred_loss,
                      'task_adv_loss': task_adv_loss})

    # def validation_step(self, batch, batch_idx):
    #     # parse batch data
    #     task_id = batch['task_id'][0].item()
    #     covar_name = self.task_id_class_label.int2str(task_id)
    #     encoded_input = {key: batch[key] for key in self.bert_encode_keys}

    #     covar_prediction, task_prediction = self.model(
    #         covar_name=covar_name, **encoded_input)
    #     covar_metric_func = self._choose_metric_func(covar_name=covar_name)

    #     valid_result = covar_metric_func(covar_prediction, batch['label'])
    #     # task_result = self._compute_task_adversarial_metric(
    #     #     task_prediction, batch['task_id'])

    #     # return {covar_name: valid_result, 'task_adv': task_result}
    #     return {covar_name: valid_result}

    # def validation_epoch_end(self, validation_step_outputs):
    #     covar_name2outputs = defaultdict(list)
    #     for out_dict in validation_step_outputs:
    #         for covar_name, out in out_dict.items():
    #             covar_name2outputs[covar_name].append(out)

    #     covar_name2valid_result = {}
    #     for covar, outputs in covar_name2outputs.items():
    #         if covar in self.continuous_covars:
    #             # R-square: higher is better
    #             square_diffs = sum([square_diff for square_diff, _ in outputs])
    #             targets = sum([targets for _, targets in outputs])
    #             valid_result = 1 - (square_diffs / targets)
    #         elif covar in self.discrete_covar2class_num:
    #             accuracy = torch.cat([accuracy for accuracy in outputs], dim=0)
    #             valid_result = torch.mean(accuracy)
    #         elif covar == 'task_adv':
    #             accuracy = torch.cat([accuracy for accuracy in outputs], dim=0)
    #             valid_result = torch.mean(accuracy)
    #         else:
    #             raise NotImplementedError
    #         covar_name2valid_result[covar] = valid_result

    #     self.log_dict(covar_name2valid_result)

    def on_fit_end(self):
        self.model.bert.backbone.save_pretrained(
            save_directory=self.save_pretrain_dir)

    def _choose_loss_func(self, covar_name):
        if covar_name in self.continuous_covars:
            return self._compute_continuous_covar_loss
        elif covar_name in self.discrete_covar2class_num:
            return self._compute_discrete_covar_loss
        else:
            raise NotImplementedError

    @staticmethod
    def _compute_continuous_covar_loss(pred, target):
        return F.mse_loss(torch.squeeze(pred, dim=-1), target, reduction='mean')

    @staticmethod
    def _compute_discrete_covar_loss(pred, target):
        log_softmax = F.log_softmax(pred, dim=-1)
        return -torch.mean(target * log_softmax)

    @staticmethod
    # lower mean adv_head better
    def _compute_task_adversarial_loss(pred, target):
        return F.cross_entropy(pred, target, reduction='mean')

    def _choose_metric_func(self, covar_name):
        if covar_name in self.continuous_covars:
            return self._compute_continuous_covar_metric
        elif covar_name in self.discrete_covar2class_num:
            return self._compute_discrete_covar_metric
        else:
            raise NotImplementedError

    @staticmethod
    def _compute_discrete_covar_metric(pred, target):
        multi_hot = torch.zeros_like(target)
        multi_hot[torch.nonzero(target)] = 1
        accuracy = pred * target
        accuracy = torch.sum(accuracy, dim=-1)
        return accuracy

    @staticmethod
    def _compute_task_adversarial_metric(pred, target):
        _, index = torch.max(pred, dim=-1)
        return index == target

    @staticmethod
    def _compute_continuous_covar_metric(pred, target):
        square_diff = torch.square(target - torch.squeeze(pred, dim=-1))
        return torch.sum(square_diff), torch.sum(target)
