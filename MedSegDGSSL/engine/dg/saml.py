from turtle import forward
import torch
import numpy as np
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from torch.nn import functional as F
import torch.nn as nn
from scipy import ndimage
from MedSegDGSSL.network import build_network
from MedSegDGSSL.utils import load_pretrained_weights, count_num_param
from MedSegDGSSL.optim import build_optimizer, build_lr_scheduler

import pdb



@TRAINER_REGISTRY.register()
class SAML(TrainerX):
    """
    Shape-aware meta-learning (SAML)

    https://arxiv.org/abs/2007.02035
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.intermediate_outputs = {}

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output
        return hook

    def build_model(self):

        cfg = self.cfg

        # build embedding net
        print("Building Embedding Network")
        self.EmbeddingNet = Embedding()
        self.EmbeddingNet.to(self.device)
        self.optim_E = build_optimizer(self.EmbeddingNet, cfg.OPTIM)
        self.sched_E = build_lr_scheduler(self.optim_E, cfg.OPTIM)
        self.register_model("EmbeddingNet", self.EmbeddingNet, self.optim_E, self.sched_E)

        return super().build_model()

    def generate_meta_train_test(self):
        num_source_domain  = len(self.cfg.DATASET.SOURCE_DOMAINS)
        num_meta_train = 2
        num_meta_test = 1

        # randomly choosing meta-train and meta-test domains
        task_list = np.random.permutation(num_source_domain)
        meta_train_index_list = task_list[:num_meta_train]
        meta_test_index_list = task_list[-num_meta_test:]
        print(
            "{} source domains in total\n \
            sample domain {} and {} as meta train set\n \
            domain {} as meta test set".format(num_source_domain, 
            meta_train_index_list[0], meta_train_index_list[1], meta_test_index_list[0])
            )

        return num_source_domain, meta_train_index_list, meta_test_index_list




    def forward_backward(self, batch):
        metatrain1_input, metatrain2_input, metatest_input, \
        metatrain1_label, metatrain2_label, metatest_label, \
        metainput_group, contour_group, metric_label_group = self.parse_batch_train(batch)


        #############
        # Conventional task on meta-train
        #############
        metatrain1_output = self.model(metatrain1_input)
        loss_metatrain1 = self.loss_func(metatrain1_output, metatrain1_label)

        metatrain2_output = self.model(metatrain2_input)
        loss_metatrain2 = self.loss_func(metatrain2_output, metatrain2_label)
        self.model_backward_and_update((loss_metatrain1+loss_metatrain2)/2, 'model')

        #############
        # Cross domain learning task on meta-test
        #############
        metatest_output = self.model(metatest_input)
        [B,N,H,W] = metatest_output.shape 

        seg_loss = self.loss_func(metatest_output, metatest_label)
        compactness_loss = self.get_compactness_loss(metatest_output.view(B,H,W,N), F.one_hot(metatest_label).squeeze())

        # get smoothness loss
        _ = self.model(metainput_group)
        embeddings1 = self.model.features["16"][0]
        resize_module = nn.Upsample(size=(H,W), mode='bilinear')
        embeddings2 = resize_module(self.model.features["32"][0])
        embeddings = torch.cat((embeddings1, embeddings2), dim=1)
        smoothness_loss = self.get_smoothness_loss(contour_group, embeddings.view(embeddings.shape[0],H,W,48), metric_label_group)
        self.model_backward_and_update(smoothness_loss, 'EmbeddingNet')

        # metaloss
        meta_loss = seg_loss + compactness_loss + 5e-3*smoothness_loss
        self.model_backward_and_update(meta_loss, 'model')



        # return loss
        loss_summary = {
            'loss_metatrain1': loss_metatrain1.item(),
            'loss_metatrain2': loss_metatrain2.item(),
            'loss_meta': meta_loss.item(),
            'loss_metatest_seg': seg_loss.item(),
            'loss_metatest_smooth': smoothness_loss.item(),
            'loss_metatest_compact': compactness_loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def get_smoothness_loss(self, contour_group, embeddings, metric_label_group):
        contour_embeddings = self.extract_contour_embedding(contour_group, embeddings)
        metric_embeddings = self.EmbeddingNet(contour_embeddings)
        loss = TripletSemihardLoss()

        return loss(embeddings=metric_embeddings, target=metric_label_group[...,0], margin=10.0)


    def parse_batch_train(self, batch):

        # get batch with 5*32 data each from different domain
        input = batch['data']
        label = batch['seg']
        domain = batch['domain']
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        # sample two meta train domains and one meta test domain
        num_source_domain, meta_train_domains, meta_test_domains = self.generate_meta_train_test()
        metatrain1_input = input[np.where(batch['domain']==meta_train_domains[0])[0]]
        metatrain2_input = input[np.where(batch['domain']==meta_train_domains[1])[0]]
        metatest_input   = input[np.where(batch['domain']==meta_test_domains[0])[0]]
        metatrain1_label = label[np.where(batch['domain']==meta_train_domains[0])[0]]
        metatrain2_label = label[np.where(batch['domain']==meta_train_domains[1])[0]]
        metatest_label   = label[np.where(batch['domain']==meta_test_domains[0])[0]]      
        assert metatrain1_input.shape[0] == self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE / num_source_domain

        metainput_group = torch.cat((metatrain1_input[:2], metatrain2_input[:1], metatest_input[:2]), 0)
        metalabel_group = torch.cat((metatrain1_label[:2], metatrain2_label[:1], metatest_label[:2]), 0)
        
        contour_group, metric_label_group = self._get_contour_sample(F.one_hot(metalabel_group).squeeze().cpu().numpy())


        return  metatrain1_input, metatrain2_input, metatest_input,\
                metatrain1_label, metatrain2_label, metatest_label,\
                metainput_group, contour_group.to(self.device), metric_label_group.to(self.device)


    def get_compactness_loss(self, pred, gt):
        """
        pred, gt: BNHW, where N is num_class
        """

        epsilon = 1e-8
        w = 0.01
        pred = pred[:,...,1]
        gt = gt[:,...,1]

        # gradient on x and y directions
        x = pred[:,1:,:] - pred[:,:-1,:]
        y = pred[:,:,1:] - pred[:,:,:-1]
        
        # perimeter length P
        delta_x = x[:,:,1:]**2
        delta_y = y[:,1:,:]**2
        length = w * torch.sqrt(delta_x + delta_y + epsilon).view(pred.shape[0],-1).sum(dim=1)

        # area A
        area = pred.view(pred.shape[0],-1).sum(dim=1)

        # loss = P^2 / 4pi*A
        loss = torch.sum((length ** 2) / (4 * area * torch.pi))

        return loss

    def _get_contour_sample(self, y_true):
        """
        y_true: BxHxWx2
        """
        
        positive_mask = np.expand_dims(y_true[..., 1], axis=3)
        metrix_label_group = np.expand_dims(np.array([1, 0, 1, 1, 0]), axis = 1)
        contour_group = np.zeros(positive_mask.shape)

        for i in range(positive_mask.shape[0]):
            slice_i = positive_mask[i]

            if metrix_label_group[i] == 1:
                # generate contour mask
                erosion = ndimage.binary_erosion(slice_i[..., 0], iterations=1).astype(slice_i.dtype)
                sample = np.expand_dims(slice_i[..., 0] - erosion, axis = 2)

            elif metrix_label_group[i] == 0:
                # generate background mask
                dilation = ndimage.binary_dilation(slice_i, iterations=5).astype(slice_i.dtype)
                sample = dilation - slice_i 

            contour_group[i] = sample
        return torch.tensor(contour_group), torch.tensor(metrix_label_group)


    def extract_contour_embedding(self, contour, embeddings):
        pdb.set_trace()

        contour_embeddings = contour * embeddings
        average_embeddings = contour_embeddings.view(contour_embeddings.shape[0],-1,48).sum(dim=1) / contour.view(contour.shape[0],-1).sum(dim=1)

        return average_embeddings


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(48, 24)
        self.fc2 = nn.Linear(24, 16)
    
    def forward(self, input):
        out = F.leaky_relu(self.fc1(input))
        out = F.leaky_relu(self.fc2(input))

        return out



def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()


class TripletSemihardLoss(nn.Module):
    """
    the same with tf.triplet_semihard_loss
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self):
        super(TripletSemihardLoss, self).__init__()

    def masked_maximum(self, data, mask, dim=1):
        """Computes the axis wise maximum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the maximum.
            Returns:
              masked_maximums: N-D `Tensor`.
                The maximized dimension is of size 1 after the operation.
            """
        axis_minimums = torch.min(data, dim, keepdim=True).values
        masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values + axis_minimums
        return masked_maximums

    def masked_minimum(self, data, mask, dim=1):
        """Computes the axis wise minimum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the minimum.
            Returns:
              masked_minimums: N-D `Tensor`.
                The minimized dimension is of size 1 after the operation.
            """
        axis_maximums = torch.max(data, dim, keepdim=True).values
        masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values + axis_maximums
        return masked_minimums

    def pairwise_distance(self, embeddings, squared=True):
        pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                     torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                     2.0 * torch.matmul(embeddings, embeddings.t())

        error_mask = pairwise_distances_squared <= 0.0
        if squared:
            pairwise_distances = pairwise_distances_squared.clamp(min=0)
        else:
            pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

        pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

        num_data = embeddings.shape[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(cudafy(torch.ones([num_data])))
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
        return pairwise_distances

    def forward(self, embeddings, target, margin=1.0, squared=True):
        """
        :param features: [B * N features]
        :param target: [B]
        :param square: if the distance squared or not.
        :return:
        """
        lshape = target.shape
        assert len(lshape) == 1
        labels = target.int().unsqueeze(-1)  # [B, 1]
        pdist_matrix = self.pairwise_distance(embeddings, squared=squared)

        adjacency = labels == torch.transpose(labels, 0, 1)

        adjacency_not = ~adjacency
        batch_size = labels.shape[0]

        # Compute the mask

        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

        mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(
            torch.transpose(pdist_matrix, 0, 1), [-1, 1]))

        mask_final = torch.reshape(torch.sum(mask.float(), 1, keepdim=True) >
                                   0.0, [batch_size, batch_size])
        mask_final = torch.transpose(mask_final, 0, 1)

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        # negatives_inside: largest D_an.
        negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = adjacency.float() - torch.diag(cudafy(torch.ones([batch_size])))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.div(torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)), num_positives)
        
        # triplet_loss = torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0))
        return triplet_loss