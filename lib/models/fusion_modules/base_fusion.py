import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class BaseFusion(nn.Module):

    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE #512
        txt_input_size = cfg.TXT_INPUT_SIZE #300
        txt_hidden_size = cfg.TXT_HIDDEN_SIZE #512
        #self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        # nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size, #else문에 들어감
                                       #num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True) #3,False
                                       #nn.LSTM(input dimension=300, output layer dimension=512, layer count=3) #map이랑 차원수 맞춰줘야되니까
        self.tex_linear = nn.Linear(txt_hidden_size, hidden_size) #512,512
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1) #512,512 

    def forward(self,textual_input, textual_mask, map_h, map_mask):
        
        #self.textual_encoder.flatten_parameters() 
        
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask 
        #torch.Size([batch_size, max_n_words, 512]) 
        txt_h = torch. stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)]) 
        #torch.Size([batch_size, 512])
        txt_h = self.tex_linear(txt_h)[:,:,None,None] 
        #torch.Size([batch_size, 512, 1, 1]) 
        '''

        txt_h=self.bert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask= attention_mask)
        txt_h=txt_h["last_hidden_state"][:,0,:]
        txt_h = self.tex_linear(txt_h)[:,:,None,None] #torch.Size([batch_size, 512, 1, 1]) 
        '''

        map_h = self.vis_conv(map_h)

        fused_h = F.normalize(txt_h * map_h) * map_mask #hardamard product
        return fused_h

