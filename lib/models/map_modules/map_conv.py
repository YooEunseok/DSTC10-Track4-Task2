from torch import nn
import torch
import torch.nn.functional as F
from models.map_modules import get_padded_mask_and_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE #512
        hidden_sizes = cfg.HIDDEN_SIZES #[512, 512, 512, 512, 512, 512, 512, 512]
        kernel_sizes = cfg.KERNEL_SIZES #[5, 5, 5, 5, 5, 5, 5, 5]
        strides = cfg.STRIDES #[1, 1, 1, 1, 1, 1, 1, 1]
        paddings = cfg.PADDINGS #[16, 0, 0, 0, 0, 0, 0, 0]
        dilations = cfg.DILATIONS #[1, 1, 1, 1, 1, 1, 1, 1]
        self.convs = nn.ModuleList()

        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes #512가 하나 append됨

        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            
            #layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            #                     kernel_size=kernel_size, stride=stride, padding=padding,
            #                     bias=bias)]
            cbr=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
            #layers += [nn.BatchNorm2d(num_features=out_channels)]

            #cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=512, out_channels=1024)
        self.enc1_2 = CBR2d(in_channels=1024, out_channels=1024)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=1024, out_channels=2048)
        self.enc2_2 = CBR2d(in_channels=2048, out_channels=2048)

        self.dec2_2 = CBR2d(in_channels=2048, out_channels=2048)
        self.dec2_1 = CBR2d(in_channels=2048, out_channels=1024)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 1024, out_channels=1024)
        self.dec1_1 = CBR2d(in_channels=1024, out_channels=1024)
        
        self.fc = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)


        #########################################################################################################
 
    def forward(self, x, mask): #torch.Size([batch_size, 512, 16, 16]) , torch.Size([batch_size, 1, 16, 16])
        padded_mask = mask       
        '''
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x)) 
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred) #torch.Size([batch_size, 1, 16, 16]), conv layer
            x = x * masked_weight
        print(x.shape)
        print(x[0])
        assert 0
        #assert 0
        '''
        #########################################################################################################
        enc1_1 = F.relu(self.enc1_1(x))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.enc1_1) #torch.Size([batch_size, 1, 16, 16]), conv layer
        enc1_1 = enc1_1 * masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(enc1_1.shape)
        #print()
        
        enc1_2 = F.relu(self.enc1_2(enc1_1))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.enc1_2) #torch.Size([batch_size, 1, 16, 16]), conv layer
        enc1_2 = enc1_2 * masked_weight
        print(padded_mask.shape)
        print(masked_weight.shape)
        print(enc1_2.shape)
        print()

        pool1 = F.relu(self.pool1(enc1_2))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.pool1, 2) #torch.Size([batch_size, 1, 16, 16]), conv layer
        print(masked_weight[0])
        pool1 = pool1 * masked_weight

        print(pool1.shape)
        print()


        enc2_1 = F.relu(self.enc2_1(pool1))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.enc2_1) #torch.Size([batch_size, 1, 16, 16]), conv layer
        print(padded_mask.shape)
        print(masked_weight.shape)
        enc2_1 = enc2_1 * masked_weight
        
        #print(enc2_1.shape)
        #print()
        
        enc2_2 = F.relu(self.enc2_2(enc2_1))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.enc2_2) #torch.Size([batch_size, 1, 16, 16]), conv layer
        enc2_2 = enc2_2 * masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(enc2_2.shape)
        #print()
        

        dec2_2 = F.relu(self.dec2_2(enc2_2))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.dec2_2) #torch.Size([batch_size, 1, 16, 16]), conv layer
        dec2_2 = dec2_2* masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(dec2_2.shape)
        #print()
        
        dec2_1 = F.relu(self.dec2_1(dec2_2))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.dec2_1) #torch.Size([batch_size, 1, 16, 16]), conv layer
        dec2_1 = dec2_1* masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(dec2_1.shape)
        #print()
        
        unpool1 = F.relu(self.unpool1(dec2_1))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.unpool1,2,5) #torch.Size([batch_size, 1, 16, 16]), conv layer
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        unpool1 = unpool1* masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(unpool1.shape)
        #print()
        

        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        #print(cat1.shape)
        #print()


        dec1_2 = F.relu(self.dec1_2(cat1))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.dec1_2) #torch.Size([batch_size, 1, 16, 16]), conv layer
        dec1_2 = dec1_2* masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(dec1_2.shape)
        #print()
        
        dec1_1 = F.relu(self.dec1_1(dec1_2))
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.dec1_1) #torch.Size([batch_size, 1, 16, 16]), conv layer
        dec1_1 = dec1_1* masked_weight
        #print(padded_mask.shape)
        #print(masked_weight.shape)
        #print(dec1_1.shape)
        #print()
        
        x = self.fc(dec1_1)
        padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.fc) #torch.Size([batch_size, 1, 16, 16]), conv layer
        x =x* masked_weight
        #print(x.shape)

        assert 0

        return x


