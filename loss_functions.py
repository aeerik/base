import torch.nn.functional as F

def custom_loss(logits, targets, pad_index=-1):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    mask = (targets != pad_index).float()
    masked_loss = loss * mask
    
    average_loss = masked_loss.sum() / mask.sum()
    
    return average_loss