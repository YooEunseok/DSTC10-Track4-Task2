""" Dataset loader for the Charades-STA dataset """
import os
import csv

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config

class Charades(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>']) #####
    vocab.stoi['<unk>'] = vocab.vectors.shape[0] #####
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0) #torch.Size([400001, 300])

    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE #vgg_rgb
        self.data_dir = config.DATA_DIR #./data/Charades-STA
        self.split = split #mode

        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            #id,subject,scene,quality,relevance,verified,script,objects,descriptions,actions,length
            reader = csv.DictReader(f) #####
            for row in reader:
                self.durations[row['id']] = float(row['length']) #비디오id-총길이

        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)),'r')
        #ex)AO8RW 0.0 6.9##a person is putting a book on a shelf.
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##") #ex)AO8RW 0.0 6.9 / a person is putting a book on a shelf.
            sent = sent.split('.\n')[0] #ex)a person is putting a book on a shelf (마침표 제거)

            vid, s_time, e_time = anno.split(" ") #ex)AO8RW / 0.0 / 6.9
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid]) #e_time과 총길이중 더 짧은것
            if s_time < e_time: #시간정보에 오류가 없는지 확인 
                annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
                #ex)[{'video': 'AO8RW', 'times': [0.0, 6.9], 'description': 'a person is putting a book on a shelf', 'duration': 33.67},.....]
        anno_file.close()
        self.annotations = annotations

    def __getitem__(self, index):
        video_id = self.annotations[index]['video'] #video id
        gt_s_time, gt_e_time = self.annotations[index]['times'] #timestamp
        description = self.annotations[index]['description'] #query
        duration = self.durations[video_id] #총길이

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()], dtype=torch.long) 
        #print(description) #ex) person they take a photograph from a box
        #print(word_idxs) #ex) tensor([ 899,   39,  190,    7, 7852,   25,    7, 1930])

        word_vectors = self.word_embedding(word_idxs) #torch.Size([8, 300])

        visual_input, visual_mask = self.get_video_features(video_id) #torch.Size([n_frames, 4096]), torch.Size([n_frames, 1])

        # Time scaled to fixed size
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
        # visual_input = interpolate_to_fixed_length(visual_input)
        
        visual_input = average_to_fixed_length(visual_input) #torch.Size([256, 4096])
        
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE #256/16=16 #target.stride가 뭘 의미하는거지

        s_times = torch.arange(0,num_clips).float()*duration/num_clips
        # [0, ... ,16]을 [0, ... ,1]로 바꿔주고 duration을 곱해줌
        # clip 시작점을 총 16개로 나눔
        # ex) tensor([ 0.0000,  1.9450,  3.8900,  5.8350,  7.7800,  9.7250, 11.6700, 13.6150,
        #     15.5600, 17.5050, 19.4500, 21.3950, 23.3400, 25.2850, 27.2300, 29.1750])
        # torch.Size([16])

        e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
        # [1, ... ,17]을 [1/16, ... ,17/16]으로 바꿔주고 duration을 곱해줌
        # clip 끝점을 총 16개로 나눔
        # ex) tensor([ 1.9450,  3.8900,  5.8350,  7.7800,  9.7250, 11.6700, 13.6150, 15.5600,
        #     17.5050, 19.4500, 21.3950, 23.3400, 25.2850, 27.2300, 29.1750, 31.1200])
        # torch.Size([16])
        
        overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                    e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips) #s,e쌍 별로 gt와 겹치는 부분 계산

        gt_s_idx = np.argmax(overlaps)//num_clips #ex) 13
        gt_e_idx = np.argmax(overlaps)%num_clips #ex) 0015

        item = {
            'visual_input': visual_input, #torch.Size([256, 4096])
            'anno_idx': index, 
            'word_vectors': word_vectors, #torch.Size([n_words, 300])
            'txt_mask': torch.ones(word_vectors.shape[0], 1), #torch.Size([n_words, 1])
            'map_gt': torch.from_numpy(overlaps), #16*16
            'reg_gt': torch.tensor([gt_s_idx, gt_e_idx]),
            'duration': duration
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        hdf5_file = h5py.File(os.path.join(self.data_dir, '{}_features.hdf5'.format(self.vis_input_type)), 'r') 
        features = torch.from_numpy(hdf5_file[vid][:]).float() #해당하는 id의 비디오의 feature를 가져와서 numpy에서 torch로 변환 

        if config.DATASET.NORMALIZE: 
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_target_weights(self):
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        pos_count = [0 for _ in range(num_clips)]
        total_count = [0 for _ in range(num_clips)]
        pos_weight = torch.zeros(num_clips, num_clips)
        for anno in self.annotations:
            video_id = anno['video']
            gt_s_time, gt_e_time = anno['times']
            duration = self.durations[video_id]
            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                        e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
            overlaps[overlaps >= 0.5] = 1
            overlaps[overlaps < 0.5] = 0
            for i in range(num_clips):
                s_idxs = list(range(0, num_clips - i))
                e_idxs = [s_idx + i for s_idx in s_idxs]
                pos_count[i] += sum(overlaps[s_idxs, e_idxs])
                total_count[i] += len(s_idxs)

        for i in range(num_clips):
            s_idxs = list(range(0, num_clips - i))
            e_idxs = [s_idx + i for s_idx in s_idxs]
            # anchor weights
            # pos_weight[s_idxs,e_idxs] = pos_count[i]/total_count[i]
            # global weights
            pos_weight[s_idxs, e_idxs] = sum(pos_count) / sum(total_count)


        return pos_weight
