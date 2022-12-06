import torch

# for the numerical issue of dice
eps = 1e-5

def compute_dice(output, target):
    """Computes the Dice over the output and predict value

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes, hwd).
        target (torch.LongTensor): ground truth labels with shape (batch_size, num_classes, hwd).

    Returns:
        dice coefficient.
    """

    output_onehot = torch.zeros_like(output)
    output_onehot.scatter_(dim=1, index=torch.argmax(output, dim=1, keepdim=True), value=1)
    output_onehot = torch.flatten(output_onehot, start_dim=2)
    
    label_onehot = torch.zeros_like(output)
    label_onehot.scatter_(dim=1, index=target.to(torch.long), value=1)
    label_onehot = torch.flatten(label_onehot, start_dim=2)

    dice_value = (2*torch.sum(output_onehot*label_onehot, dim=[0, 2])+eps) / (torch.sum((label_onehot + output_onehot), dim=[0, 2]) + eps)
    return dice_value
