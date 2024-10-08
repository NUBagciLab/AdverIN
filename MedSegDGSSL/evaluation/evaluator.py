import json
import os.path as osp
from collections import OrderedDict, defaultdict

import torch
import pandas as pd
import numpy as np
from skimage import transform
from sklearn.metrics import confusion_matrix

from .build import EVALUATOR_REGISTRY
from MedSegDGSSL.metrics.accuracy import compute_dice
from MedSegDGSSL.evaluation.case_evaluate import evaluate_single_case, default_metrics
from MedSegDGSSL.utils.writter import write_2d_image, write_3d_image
from MedSegDGSSL.utils.tools import mkdir_if_missing


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self.best_acc = 0
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matched = pred.eq(gt).float()
        self._correct += int(matched.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i].item())
                self._per_class_res[label].append(matched_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100. * self._correct / self._total
        self.best_acc = max(self.best_acc, acc)
        err = 100. - acc
        results['accuracy'] = acc
        results['error_rate'] = err
        results['best_accuracy'] = self.best_acc

        print(
            '=> result\n'
            '* total: {:,}\n'
            '* correct: {:,}\n'
            '* current accuracy: {:.2f}%\n'
            '* best accuracy: {:.2f}%\n'
            '* error: {:.2f}%'.format(self._total, self._correct, acc, self.best_acc, err)
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print('=> per-class result')
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100. * correct / total
                accs.append(acc)
                print(
                    '* class: {} ({})\t'
                    'total: {:,}\t'
                    'correct: {:,}\t'
                    'acc: {:.2f}%'.format(
                        label, classname, total, correct, acc
                    )
                )
            print('* average: {:.2f}%'.format(np.mean(accs)))

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize='true'
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, 'cmat.pt')
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results


@EVALUATOR_REGISTRY.register()
class Segmentation(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self.num_classes  = len(self._lab2cname)

        self.average_dice = 0
        self.best_dice = 0
        self._dice_class = 0
        self._dice_list = []

    def reset(self):
        self._dice = 0
        self._dice_class = 0

    def process(self, mo, gt):
        dice_value = compute_dice(mo, gt)
        self._dice_list.append(dice_value.data.cpu().numpy())

    def evaluate(self):
        results = OrderedDict()
        dice = 100. * np.mean(np.stack(self._dice_list), axis=0)
        self._dice_class = dice
        self.average_dice = float(dice[1]) if self.num_classes==2 else float(np.mean(dice[1:]))
        self.best_dice = max(self.best_dice, self.average_dice)
        err = 100. - self.average_dice
        results['dice'] = self.average_dice
        results['error_rate'] = err
        results['best_dice'] = self.best_dice

        print(
            '=> result\n'
            '* current dice: {:.2f}\n'
            '* best dice: {:.2f}\n'
            '* error: {:.2f}'.format(self.average_dice, self.best_dice, err)
        )

        return results



@EVALUATOR_REGISTRY.register()
class FinalSegmentation(EvaluatorBase):
    """Evaluator for classification.
    Need to add domain support here
    """

    def __init__(self, cfg, lab2cname, data_shape="3D", metrics=default_metrics, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self.metrics = metrics
        self._data_shape = data_shape
        self.output_dir = cfg.OUTPUT_DIR
        self.output_dir_summary = osp.join(self.output_dir, "summary")
        self.num_classes  = len(self._lab2cname)

        # This evaluation should be based on domain
        self._mean_evaludation_dict = {}
        self._evaluation_dict = {}
        self.exp_distance = "Distance"

        self.writter = self.get_writer()

    def reset(self):
        self._mean_evaludation_dict = {}
        self._evaluation_dict = {}

    def process(self, mo, gt, case_meta):
        case_name = case_meta['case_name']
        case_spacing = case_meta['spacing']
        case_orgsize = case_meta['org_size']
        domain = case_meta["domain"]
        predict_np, label_np = mo.data.cpu().numpy(), gt.data.cpu().numpy()
        predict_np, label_np = np.argmax(predict_np, axis=1), label_np[:, 0]

        # For squeezing the batch size direction if needed
        if predict_np.shape[0] == 1:
            predict_np, label_np = predict_np[0], label_np[0]
        # print('Before', predict_np.shape)
        predict_np, label_np = transform.resize(predict_np, case_orgsize, order=0), transform.resize(label_np, case_orgsize, order=0)
        # print(np.sum(predict_np==1), np.sum(label_np==1), label_np.shape)
        evaluation_value = evaluate_single_case(predict_np, label_np,
                                                case_name=case_name, voxel_spacing=case_spacing,
                                                labels=self._lab2cname, metric_list=self.metrics)
        temp_result_dict = {case_name: evaluation_value}
        # print(temp_result_dict)
        if domain not in self._evaluation_dict.keys():
            self._evaluation_dict[domain] = {}
        self._evaluation_dict[domain].update(temp_result_dict)

        ## write image here
        predict_folder = osp.join(self.output_dir, "out_image", domain, "prediction")
        label_folder = osp.join(self.output_dir, "out_image", domain, "label")
        mkdir_if_missing(predict_folder)
        mkdir_if_missing(label_folder)
        self.writter(predict_np, out_dir=predict_folder, 
                     case_name=case_name, meta_info=case_meta)
        self.writter(label_np, out_dir=label_folder, 
                     case_name=case_name, meta_info=case_meta)
    
    def get_writer(self):
        ### use data modality to distinguish
        if self._data_shape == "3D":
            return write_3d_image
        else:
            return write_2d_image

    def evaluate(self, extra_name:str=''):
        mkdir_if_missing(self.output_dir_summary)

        for domain in self._evaluation_dict.keys():
            results = {}
            for label in list(self._lab2cname.values()):
                self._mean_evaludation_dict[label] = {}
                results[label] = {}
                for metric in self.metrics:
                    temp_metric_list = [item[label][metric] for item in (self._evaluation_dict[domain].values())]
                    temp_mean = np.mean(temp_metric_list)
                    temp_std = np.std(temp_metric_list)
                    self._mean_evaludation_dict[label][metric] = {}
                    self._mean_evaludation_dict[label][metric]["mean"] = temp_mean
                    self._mean_evaludation_dict[label][metric]["std"] = temp_std
                    if self.exp_distance in metric:
                        results[label][metric] = f"{np.round(temp_mean, 2)} " + "\u00B1" + f" {np.round(temp_std, 2)}"
                    else:
                        results[label][metric] = f"{np.round(temp_mean, 4)} " + "\u00B1" + f" {np.round(temp_std, 4)}"

            pf = pd.DataFrame.from_dict(results, orient='index')
            print('=> result\n', pf)

            ### save the summary result
            print(self.output_dir_summary, domain)
            pf.to_csv(osp.join(self.output_dir_summary, f'{domain}_summary_result{extra_name}.csv'))
            with open(osp.join(self.output_dir_summary, f'{domain}_detail_result{extra_name}.json'), 'w') as f:
                json.dump({"case_level": self._evaluation_dict[domain],
                           "mean_level": self._mean_evaludation_dict}, f, indent=4)
