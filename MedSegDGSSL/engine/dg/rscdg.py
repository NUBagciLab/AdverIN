import torch
from torch.nn import functional as F

from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX

from MedSegDGSSL.metrics import compute_dice
from MedSegDGSSL.network.ops.gradient_challenge.rsc import RSC


@TRAINER_REGISTRY.register()
class RSCDG(TrainerX):
    """RSC baseline."""
    def build_model(self):
        self.rsc = RSC(challenge_list=self.cfg.MODEL.RSC.CHALLENGE_LIST,
                       percentile=self.cfg.MODEL.RSC.PERCENTILE)
        print('Building Segmentation network')
        assert 'encdec' in self.cfg.MODEL.NAME, \
                'For RSC should use encoder-decoder version'
        super().build_model()

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        with torch.no_grad():
            features = self.model.encoder(input)
        features = self.rsc.wrap_variable(features)
        output = self.model.decoder(features)
        #print(input.shape, torch.sum(label))

        ## use challenge loss to backward
        label_onehot = torch.zeros_like(output)
        label_onehot.scatter_(dim=1, index=label.to(torch.long), value=1)
        label_onehot = torch.flatten(label_onehot, start_dim=2)
        output = torch.flatten(output, start_dim=2)
        challenge_loss = torch.mean(label_onehot*output)
        challenge_loss.backward()
        feature_masks = self.rsc.generate_mask(features)

        features = self.model.encoder(input)
        features = self.rsc.mask_forward(features, feature_masks)
        output = self.model.decoder(features)
        loss = self.loss_func(output, label)

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
