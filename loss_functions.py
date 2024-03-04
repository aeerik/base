import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def custom_loss(logits, targets, pad_index=-1):
    targets = targets.to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none').to(device)
    
    mask = (targets != pad_index).float()
    masked_loss = loss * mask
    
    average_loss = masked_loss.sum() / mask.sum()
    
    return average_loss