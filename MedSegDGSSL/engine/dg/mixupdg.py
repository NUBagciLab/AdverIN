import torch
from torch.nn import functional as F
import monai.losses as losses

from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.metrics import compute_dice
from MedSegDGSSL.network.ops.mixup import MixUp


def to_onehot(input:torch.Tensor, num_classes:int=2):
    """ transfer label to one hot
    """
    output_shape = list(input.shape)
    output_shape[1] = num_classes
    output = torch.zeros(size=output_shape, device=input.device)
    output.scatter_(dim=1, index=input.to(torch.long), value=1)
    return output


@TRAINER_REGISTRY.register()
class MixUpDG(TrainerX):
    """Input Augmentation via MixUp."""
    def build_model(self):
        self.mixup_ops = MixUp(preserve_order=True)
        self.mixup_ops.to(self.device)
        return super().build_model()
    
    def get_loss_func(self):
        """Get loss function, resetting due to mixup needs onehot labeling
        """
        potential_seg_loss_list = ["DiceLoss", "DiceCELoss", "DiceFocalLoss"]
        if self.cfg.LOSS in potential_seg_loss_list:
            loss = getattr(losses, self.cfg.LOSS)(include_background=True, softmax=True,
                                                  to_onehot_y=False, batch=False)
            # loss = losses.DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
        else:
            raise FileNotFoundError(f"loss type {self.cfg.LOSS} not support, only support {potential_seg_loss_list}")
        return loss

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        # direct reverse the order of samples
        label_onehot = to_onehot(label, num_classes=self.num_classes)
        input_reserve, label_reverse = torch.flip(input, dims=(0,)), torch.flip(label_onehot, dims=(0,))
        input, label_onehot = self.mixup_ops(input, input_reserve, label_onehot, label_reverse)
        output = self.model(input)
        #print(input.shape, torch.sum(label))
        loss = self.loss_func(output, label_onehot)
        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item()}
        dice_value = compute_dice(output, label)
        for i in range(self.num_classes-1):
            loss_summary[f'dice {str(i+1)}'] = dice_value[i+1].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['data']
        label = batch['seg']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
