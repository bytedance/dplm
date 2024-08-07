import torchmetrics
import torch

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import DPLMBaseModel


@register_model
class DPLMPPIModel(DPLMBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: other arguments for DPLMBaseModel
        """
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        
        hidden_size = self.model.net.config.hidden_size * 2
        classifier = torch.nn.Sequential(
                        Linear(hidden_size, hidden_size),
                        ReLU(),
                        Linear(hidden_size, 2)
                    )
        
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs_1, inputs_2):
        if self.freeze_backbone:
            hidden_1 = torch.stack(self.get_hidden_states(inputs_1, reduction="mean"))
            hidden_2 = torch.stack(self.get_hidden_states(inputs_2, reduction="mean"))
        else:
            hidden_1 = self.model(inputs_1['input_ids'], return_last_hidden_state=True)[1][:, 0, :]
            hidden_2 = self.model(inputs_2['input_ids'], return_last_hidden_state=True)[1][:, 0, :]

        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        return self.model.classifier(hidden_concat)
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train/train_loss"] = loss
            self.log_info(log_dict, on_epoch=False, on_step=True)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        print(log_dict)
        self.log_info(log_dict)

        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid/valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        self.log_info(log_dict)
        self.reset_metrics("valid")
        #self.check_save_condition(log_dict["valid/valid_acc"], mode="max")