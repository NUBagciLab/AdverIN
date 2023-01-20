import torch
from torch.nn import functional as F

import time
import datetime
from MedSegDGSSL.dataset.data_manager import DataManager
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.engine.dg.vanilla import Vanilla
from MedSegDGSSL.metrics import compute_dice, to_onehot
from MedSegDGSSL.utils.meters import AverageMeter, MetricMeter


@TRAINER_REGISTRY.register()
class DiffusionTrainer(Vanilla):
    ''' To implement intra domain training process
    use the diffusion model
    '''
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
        label = to_onehot(label, self.num_classes)
        loss = self.model.get_p_losses(input, label)
        #print(input.shape, torch.sum(label))
        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item()}

        if is_eval:
            output = self.model(input)
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

    def final_evaluation(self):
        """A generic final evaluation pipeline."""
        # extra_name=f'_fold_{self.cfg.DATASET.FOLD}'
        super().final_evaluation()
