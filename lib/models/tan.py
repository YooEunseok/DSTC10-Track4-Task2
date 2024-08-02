from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules


class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS) #FrameAvgPool
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS) #SparsePropMaxPool
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS) #BaseFusion
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS) #MapConv
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1) #512
        #self.linear=nn.Linear(128,2048)
        #self.dropout = nn.Dropout(0.15)

    def forward(self,textual_input, textual_mask,  visual_input):
        #print(visual_input)

        #print()
        #vis_h=self.linear(visual_input)
        #print('1) ',vis_h.shape)
        #vis_h = self.dropout(vis_h)
        #print(vis_h[0:10,0:20,0])
        #print()
        #print()

        #torch.Size([batch_size, 4096, 256]) 
        vis_h = self.frame_layer(visual_input.transpose(1, 2)) 
        #vis_h = self.frame_layer(vis_h.transpose(1, 2)) 
        #print('2) ',vis_h.shape)
        #torch.Size([batch_size, 512, 16])
        map_h, map_mask = self.prop_layer(vis_h)
        #print('3) ',map_h.shape)
        #print('3) ',map_mask.shape)
        #torch.Size([batch_size, 512, 16, 16]) , torch.Size([batch_size, 1, 16, 16])
        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        #print('4) ',fused_h.shape)
        #torch.Size([batch_size, 512, 16, 16])
        fused_h = self.map_layer(fused_h, map_mask) 
        #print('5) ',fused_h.shape)
        #torch.Size([batch_size, 512, 16, 16])
        prediction = self.pred_layer(fused_h) * map_mask 
        #print('6) ',prediction.shape)
        #torch.Size([batch_size, 1, 16, 16])


        return prediction, map_mask

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
