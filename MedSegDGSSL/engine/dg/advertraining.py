import torch
import numpy as np
from torch.nn import functional as F

from MedSegDGSSL.optim import build_optimizer, build_lr_scheduler
from MedSegDGSSL.utils import count_num_param
from MedSegDGSSL.network import build_network
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.metrics import compute_dice

@TRAINER_REGISTRY.register()
class AdverTraining(TrainerX):
    """Adversarial training.
    """
    def build_model(self):
        cfg = self.cfg

        print('Building Adversial Training Block')
        self.adv = build_network(cfg.MODEL.ADVER_MODEL_NAME, cfg=cfg)
        self.adv.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.adv)))
        # For adversarial training, the optimizer should be just sgd without any momentum
        self.optim_adv = torch.optim.SGD(self.adv.parameters(), lr=cfg.MODEL.ADVER_RATE, 
                                         momentum=0, weight_decay=0)
        self.register_model('adv', self.adv, self.optim_adv)

        print('Building Segmentation network')
        super().build_model()

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)

        # Compute General Attacking
        output = self.model(self.adv(input))
        loss_adver = self.loss_func(output, label)
        # Perhaps clip the grad if needed?
        # orch.nn.utils.clip_grad_norm_(self.adv.parameters())
        self.model_backward_and_update(loss_adver, 'adv')

        output = self.model(self.adv(input))
        #print(input.shape, torch.sum(label))
        loss = self.loss_func(output, label)
        self.model_backward_and_update(loss, 'model')

        self.adv.reset()
        loss_summary = {
            'loss': loss_adver.item(),
            'adver loss': loss.item()}
        dice_value = compute_dice(output, label)
        for i in range(self.num_classes-1):
            loss_summary[f'dice {str(i+1)}'] = dice_value[i+1].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr("model")

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['data']
        label = batch['seg']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def get_current_lr(self):
        name = "model"
        return self._optims[name].param_groups[0]["lr"]
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir, model_name="model")

            if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
                curr_result = self.test(split="val")
                is_best = curr_result > self.best_result
                
                if is_best:
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        is_best=is_best, 
                        val_result=curr_result,
                        model_name="model-best.pth.tar"
                    )


@TRAINER_REGISTRY.register()
class AdverHist(AdverTraining):
    """ Specify for adverhist trianing
    """
    def forward_backward(self, batch):
        input, label, region = self.parse_batch_train(batch)

        # Compute General Attacking
        # select region
        select_region = self.select_region(region)
        output = self.model(self.adv(input)*select_region + (1-select_region)*input)
        loss_adver = self.loss_func(output, label)
        # Perhaps clip the grad if needed?
        # orch.nn.utils.clip_grad_norm_(self.adv.parameters())
        self.model_backward_and_update(loss_adver, 'adv')
        # print(torch.max(self.adv.params), torch.min(self.adv.params))

        output = self.model(self.adv(input)*select_region + (1-select_region)*input)
        #print(input.shape, torch.sum(label))
        loss = self.loss_func(output, label)
        self.model_backward_and_update(loss, 'model')

        self.adv.reset()
        loss_summary = {
            'loss': loss_adver.item(),
            'adver loss': loss.item()}
        dice_value = compute_dice(output, label)
        for i in range(self.num_classes-1):
            loss_summary[f'dice {str(i+1)}'] = dice_value[i+1].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr("model")

        return loss_summary
    
    def select_region(self, region):
        random_index = np.random.randint(self.cfg.MODEL.ADVER_HIST.NUM_REGION,
                                         size=self.cfg.MODEL.ADVER_HIST.SELECT_REGION)
        mask = torch.zeros_like(region)
        for i in list(random_index):
            mask += (region==i)
        mask = (mask > 0.5).to(torch.float)
        return mask


    def parse_batch_train(self, batch):
        input = batch['data']
        label = batch['seg']
        region = batch['region']
        input = input.to(self.device)
        label = label.to(self.device)
        region = region.to(self.device)
        return input, label, region
