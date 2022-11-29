from MedSegDGSSL.dataset.build import DATASET_REGISTRY
from MedSegDGSSL.dataset.base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class ProstateMRI(DatasetBase):
    """Prostate Segmentation

    Statistics:
        - 6 domains: "BMC", "HK", "I2CVB", "UCL", "RUNMC", "BIDMC"
        - Prostate Segmentation
    """
    dataset_name = 'ProstateMRI'
    domains = ["BMC", "HK", "BIDMC"]

    def __init__(self, cfg):

        super().__init__(data_dir=cfg.DATASET.ROOT,
                         num_classes=cfg.NUM_CLASSES,
                         train_domains=cfg.DATASET.SOURCE_DOMAINS,
                         test_domains=cfg.DATASET.TARGET_DOMAINS)

if __name__ == "__main__":
    from MedSegDGSSL.config.defaults import _C
    _C.DATASET.ROOT = "/home/zze3980/project/AdverHistAug/Data/ProstateMRI/processed/train3D"
    _C.NUM_CLASSES = 2
    _C.DATASET.SOURCE_DOMAINS = ["BMC", "HK"]
    _C.DATASET.TARGET_DOMAINS = ["BIDMC"]
    prostate = ProstateMRI(_C)
