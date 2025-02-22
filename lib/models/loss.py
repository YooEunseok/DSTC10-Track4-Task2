import torch
import torch.nn.functional as F

def bce_rescale_loss(scores, masks, targets, cfg):
    #print(scores)
    #print()

    #print(targets)
    #print()
    #print()
    #print()

    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS #0.5, 1.0, 0.0
    joint_prob = torch.sigmoid(scores) * masks
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou) #(target-0.5)*2
    
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
   

    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    
    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value, joint_prob