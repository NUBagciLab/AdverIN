from MedSegDGSSL.dataset.build import DATASET_REGISTRY
from MedSegDGSSL.dataset.data_base import Datum, DatasetBase


@DATASET_REGISTRY.register()
class Cardiac(DatasetBase):
    """Cardiac Ventricle Segmentation

    Statistics:
        - 6 domains: Domain1, Domain2, Domain3, Domain4 (Domain123 coming from M&M, Domain4 coming from ACDC2017)
        - Cardiac ventricle Segmentation
    """
    dataset_name = 'Caridac'
    domains = ["Domain1", "Domain2", "Domain3", "Domain4"]
    labels = {'0': 'Background', '1':'Left ventricle', '2':'Right ventricle', '3':'Myocardium'}
    data_shape = "3D"
    def __init__(self, cfg):

        super().__init__(data_dir=cfg.DATASET.ROOT,
                         train_domains=cfg.DATASET.SOURCE_DOMAINS,
                         test_domains=cfg.DATASET.TARGET_DOMAINS)
        self._lab2cname = self.labels
        self.num_classes = len(self.labels)

