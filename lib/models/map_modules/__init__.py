import torch
import torch.nn.functional as F
import numpy as np
from pprint import pprint
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=1000)

def get_padded_mask_and_weight(*args):
    if len(args) == 2: #torch.Size([batch_size, 1, 16, 16]), conv layer
        mask, conv = args   
        print("2222222222222")

        #print(mask.shape)
        print(mask[0])
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation))

        print(masked_weight[0])
        print()
        print()
        a=input()

    elif len(args) == 3:
        
        mask, pool, s= args
        print("3333333333333")
        print(mask.shape)
        print(mask[0])
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1,2, 2).cuda(), stride=2, padding=0, dilation=1))
        
        print(masked_weight[0])
        print()
        print()
        a=input()
    
    elif len(args) == 4:
        mask, pool, e,s= args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1,3, 3).cuda(), stride=1, padding=s, dilation=1))
        print("44444444444444")
        print(mask.shape)
        print(mask[0])
        print(masked_weight[0])
        print()
        print()

    elif len(args) == 5:
        mask, k, s, p, d = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, k, k).cuda(), stride=s, padding=p, dilation=d))

    else:
        raise NotImplementedError

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0] #conv.kernel_size[0] * conv.kernel_size[1]  
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight #torch.Size([batch_size, 1, 44, 44]) ... torch.Size([batch_size, 1, 16, 16])

from .map_conv import MapConv