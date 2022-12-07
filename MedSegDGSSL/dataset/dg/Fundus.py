from MedSegDGSSL.dataset.build import DATASET_REGISTRY
from MedSegDGSSL.dataset.data_base import Datum, DatasetBase


@DATASET_REGISTRY.register()
class Fundus(DatasetBase):
    """Fundus Segmentation

    Statistics:
        - 4 domains: "Domain1", "Domain2", "Domain3", "Domain4"
        - Fundus Segmentation
    """
    dataset_name = 'Fundus'
    domains = ["Domain1", "Domain2", "Domain3", "Domain4"]
    labels = {"0": "background", "1": "Optic Disc", "2": "Optic Cup"}
    def __init__(self, cfg):

        super().__init__(data_dir=cfg.DATASET.ROOT,
                         train_domains=cfg.DATASET.SOURCE_DOMAINS,
                         test_domains=cfg.DATASET.TARGET_DOMAINS)
        self._lab2cname = self.labels
        self.num_classes = len(self.labels)

