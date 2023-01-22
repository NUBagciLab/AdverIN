from MedSegDGSSL.dataset.build import DATASET_REGISTRY
from MedSegDGSSL.dataset.data_base import Datum, DatasetBase


@DATASET_REGISTRY.register()
class ProstateMRI(DatasetBase):
    """Prostate Segmentation

    Statistics:
        - 6 domains: "BMC", "HK", "I2CVB", "UCL", "RUNMC", "BIDMC"
        - Prostate Segmentation
    """
    dataset_name = 'ProstateMRI'
    domains = ["BMC", "HK", "I2CVB", "UCL", "RUNMC", "BIDMC"]
    labels = {"0": "Background", "1": "Prostate"}
    data_shape = "3D"
    def __init__(self, cfg):

        super().__init__(data_dir=cfg.DATASET.ROOT,
                         train_domains=cfg.DATASET.SOURCE_DOMAINS,
                         test_domains=cfg.DATASET.TARGET_DOMAINS)
        self._lab2cname = self.labels
        self.num_classes = len(self.labels)

