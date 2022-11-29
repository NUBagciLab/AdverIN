from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


dice_metric = DiceMetric(include_background=False)

def compute_dice(output, target):
    """Computes the Dice over the output and predict value

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes, hwd).
        target (torch.LongTensor): ground truth labels with shape (batch_size, num_classes, hwd).

    Returns:
        dice coefficient.
    """
    num_classes = output.size(1)
    to_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)

    if isinstance(output, (tuple, list)):
        output = output[0]

    dice_value = dice_metric(y_pred=to_discrete(output), y=target)

    return dice_value
