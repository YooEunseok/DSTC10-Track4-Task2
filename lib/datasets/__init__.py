from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from core.config import config

def collate_fn(batch):

    #batch_input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
    #batch_attention_mask = torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long)
    #batch_token_type_ids = torch.tensor([b['token_type_ids'] for b in batch], dtype=torch.long)
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]

    batch_map_gt = [b['map_gt'] for b in batch]

    batch_anno_idxs = [b['anno_idx'] for b in batch]
    #print(batch)
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    #batch_ss = [b['sent'] for b in batch]
    #batch_tt = [b['timestamp'] for b in batch]

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][0,:num_clips,:num_clips] = map_gt

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,

        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        
        #'batch_input_ids': batch_input_ids,
        #'batch_attention_mask':batch_attention_mask,
        #'batch_token_type_ids':batch_token_type_ids,

        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
        #'batch_ss':batch_ss,
        #'batch_tt':batch_tt
    }

    return batch_data

def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS #256
    num_clips = visual_input.shape[0] #n_frames
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips 
    #([0,1,...,256] /256) * n_frames #0~256을 0과1사이로 만들어주고 *frames
    #한 비디오를 프레임에 맞게 같은 비율로 256개 나눠주는 작업
    # ex) tensor([  0.0000,   0.7305,   1.4609,   2.1914,   2.9219,   3.6523,   4.3828,
    #     ...
    #     184.0781, 184.8086, 185.5391, 186.2695, 187.0000])
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1)) 
    #반올림해준값이랑 n_frames-1중 작은값이 들어가게됨
    # ex) tensor([  0,   1,   1,   2,   3,   4,   4,   5,   6,   7,   7,   8,   9,   9,
    #     ...
    #     184, 185, 186, 186, 186])

    new_visual_input = []
    for i in range(num_sample_clips): #256
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx: #s<e
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0)) #두 프레임의 평균값
        else: #s=e
            new_visual_input.append(visual_input[s_idx]) #해당 프레임
    new_visual_input = torch.stack(new_visual_input, dim=0) #torch.Size([256, 4096])
    return new_visual_input

def average_to_fixed_length_audio(audio_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS #256
    num_clips = audio_input.shape[0] #n_frames
    #print('audio_num_clips: ',num_clips)
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips 

    #([0,1,...,256] /256) * n_frames #0~256을 0과1사이로 만들어주고 *frames
    #한 비디오를 프레임에 맞게 같은 비율로 256개 나눠주는 작업
    # ex) tensor([  0.0000,   0.7305,   1.4609,   2.1914,   2.9219,   3.6523,   4.3828,
    #     ...
    #     184.0781, 184.8086, 185.5391, 186.2695, 187.0000])
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1)) 
    #반올림해준값이랑 n_frames-1중 작은값이 들어가게됨
    # ex) tensor([  0,   1,   1,   2,   3,   4,   4,   5,   6,   7,   7,   8,   9,   9,
    #     ...
    #     184, 185, 186, 186, 186])


    new_audio_input = []
    for i in range(num_sample_clips): #256
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx: #s<e
            new_audio_input.append(torch.mean(audio_input[s_idx:e_idx],dim=0)) #두 프레임의 평균값
        else: #s=e
            new_audio_input.append(audio_input[s_idx]) #해당 프레임
    new_audio_input = torch.stack(new_audio_input, dim=0) #torch.Size([256, 128])
    return new_audio_input

from datasets.activitynet import ActivityNet
from datasets.charades import Charades
from datasets.tacos import TACoS
from datasets.dstc import DSTC

