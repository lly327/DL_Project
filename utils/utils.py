import torch


def _expand_onehot_labels_dice(pred, target):
    """Expand onehot labels to match the size of prediction"""
    num_classes = pred.shape[0]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target.long(), num_classes+1)
    one_hot_target = one_hot_target[..., :num_classes].permute(2,0,1)
    return one_hot_target