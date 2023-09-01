from ..build import NETWORK_REGISTRY, build_network
from .UNet import basicunet, naiveunet
from .UNet_StyleAug import basicunet_dsu, basicunet_bin, basicunet_mixstyle, basicunet_padain
from .UNet_EncDec import basicunet_encdec
from .AttentionUNet import attenionunet
from .UNet_Diff import diff_unet
from .ResUNet import resunet34, resunet34_nopretrain
