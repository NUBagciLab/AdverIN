import torch
from torch.nn import functional as F

import time
import datetime
import numpy as np
import monai.losses as losses

from contextlib import contextmanager
from MedSegDGSSL.dataset.data_manager import DataManager
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.engine.cross_fold.intra_trainer import IntraTrainer
from MedSegDGSSL.metrics import compute_dice, to_onehot
from MedSegDGSSL.utils.meters import AverageMeter, MetricMeter
from MedSegDGSSL.network.diffusion import EMA

@TRAINER_REGISTRY.register()
class IntraDiffusionTrainer(IntraTrainer):
    ''' To implement intra domain training process
    use the diffusion model
    '''
    def build_model(self):
        super().build_model()
        self.model_ema = EMA(model=self.model)
        self.register_model(name='ema', model=self.model_ema)

    @contextmanager
    def ema_scope(self, context=None):
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model)
        if context is not None:
            print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            self.model_ema.restore(self.model.parameters())
            if context is not None:
                print(f"{context}: Restored training weights")

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

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            if self.batch_idx == 0 :
                is_eval = True
            else:
                is_eval = False
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, is_eval)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch, is_eval:bool=False):
        input, label = self.parse_batch_train(batch)
        self.model_ema.copy_to(self.model)
        
        loss_summary = {}
        if is_eval:
            output = self.model(input)
            dice_value = compute_dice(output, label)

            for i in range(self.num_classes-1):
                loss_summary[f'dice {str(i+1)}'] = dice_value[i+1].item()

        label = to_onehot(label, self.num_classes)
        if np.random.random() > 0.5 or is_eval:
            label = 2*label - 1
            loss = self.model.get_p_losses(input, label)
            loss_summary['rec_loss'] = loss.item()
        else:
            loss = self.model.get_seg_losses(input, label, segloss_func=self.loss_func)
            loss_summary['seg_loss'] = loss.item()
        self.model_backward_and_update(loss, 'model')

        self.model_ema(self.model)
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['data']
        label = batch['seg']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def final_evaluation(self):
        """A generic final evaluation pipeline."""
        # extra_name=f'_fold_{self.cfg.DATASET.FOLD}'
        super().final_evaluation()
    
    @torch.no_grad()
    def test(self, split=None):
        with self.ema_scope():
            return super().test(split=split)
