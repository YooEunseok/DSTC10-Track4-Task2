
import os
import csv

import h5py
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
from collections import defaultdict
import pickle

from . import average_to_fixed_length
from . import average_to_fixed_length_audio
from core.eval import iou
from core.config import config
#from transformers import BertTokenizer, BertModel
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel

class DSTC(data.Dataset):
    

    
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>']) #####
    vocab.stoi['<unk>'] = vocab.vectors.shape[0] #####
    vocab.itos.extend(['<sep>']) #####
    vocab.stoi['<sep>'] = vocab.vectors.shape[0] #####
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim), torch.zeros(1, vocab.dim)], dim=0) #torch.Size([400001, 300])
    
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(DSTC, self).__init__()
        print('----------------',split)

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE #vgg_rgb
        self.data_dir = config.DATA_DIR #../../AVSD-DSTC10_baseline/data
        self.split = split #mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(self.split)

        self.durations = {}
        '''
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            #id,subject,scene,quality,relevance,verified,script,objects,descriptions,actions,length
            reader = csv.DictReader(f) #####
            for row in reader:
                self.durations[row['id']] = float(row['length']) #비디오id-총길이
        '''
        if self.split=='train':
            print('11111')
            self.data_file=pd.read_csv(self.data_dir+'/dstc10_'+self.split+'.csv', sep='\t')
            print(self.data_dir+'/dstc10_'+self.split+'.csv')
        else:
            print('22222')
            self.data_file=pd.read_csv(self.data_dir+'/dstc10_val.csv', sep='\t')
            #self.data_file=pd.read_csv(self.data_dir+'/dstc10_test.csv', sep='\t')

            print(self.data_dir+'/dstc10_test.csv')


        count=0

        while 1:
            try:
                video_id, _, _, _, duration, _, _ = self.data_file.iloc[count]
                self.durations[video_id] = float(duration)

                count+=1
            except:
                break      
  


        '''
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
        '''
        self.filter = lambda x : x

        gts = defaultdict(list)
        intervals = defaultdict(list)
        '''
        if self.split=='train':
            print('train!!!!!!!!!!!!!')
            gt = json.load(open(self.data_dir+'/train_set4DSTC10-AVSD.json'))
            annotations = []
            for d in gt['dialogs']:
                vid = d['image_id']
                for t, turn in enumerate(d['actions'], 1):
                    timestamp_list=[]
                    #gts['%s_res%03d' % (vid, t)].append(self.filter(turn['answer']))
                    s,e=turn['timestamp'].split(':')
                    timestamp_list.append(float(s))
                    timestamp_list.append(float(e))
                    #intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
                    #reason['timestamp'][0]=round(reason['timestamp'][0],4)
                    #reason['timestamp'][1]=round(reason['timestamp'][1],4)
                    sent=self.filter(turn['action'])
                    if timestamp_list[0]<timestamp_list[1]:
                        
                        annotations.append({'video':vid, 'times':timestamp_list, 'description': sent, 'duration': self.durations[vid]})
            self.annotations = annotations #18565개가 있음 *****
        '''
        temp='hello'
        temp1='hello'

        if self.split=='train':
            data = {}
            #print('train!!!!!!!!!!!!!')
            gt = json.load(open(self.data_dir+'/train_set4DSTC8-AVSD+reason.json'))
            #gt = json.load(open(self.data_dir+'/train_set4DSTC10-AVSD.json'))
            train_count1=0
            train_count2=0
            train_count3=0

            file=open('../../../hdd1/AST/audioset_class.dict',"rb") 
            content=pickle.load(file) #class-index
            index_dict = {j:i for i,j in content.items()} #index-class


            annotations = []
            for d in gt['dialogs']:
                vid = d['image_id']
                #print(vid)
                for t, turn in enumerate(d['dialog'], 1):
                    #gts['%s_res%03d' % (vid, t)].append(self.filter(turn['answer']))
                    if 'reason' in turn:
                        for reason in turn['reason']:
                            #intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
                            #reason['timestamp'][0]=round(reason['timestamp'][0],4)
                            #reason['timestamp'][1]=round(reason['timestamp'][1],4)
                            sent=self.filter(turn['question'])+" [SEP] "+self.filter(turn['answer'])
                            #sent=self.filter(turn['answer'])


                            if reason['timestamp'][0]<reason['timestamp'][1]:
                                if reason['timestamp'][0]<2:
                                    train_count1=train_count1+1
                                if self.durations[vid]>=reason['timestamp'][1]>self.durations[vid]-2:
                                    train_count2=train_count2+1
                                if reason['timestamp'][0]<2 and self.durations[vid]>=reason['timestamp'][1]>self.durations[vid]-2:
                                    train_count3=train_count3+1
                                #annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})
                                audio_keyword=["audio","audible","noise","sound","hear anything",
                                "background sound","any noise","any audio","any sound","can you hear",
                                "do you hear","speak","talk","conversation","say anything","saying","dialogue",
                                "bark","meow","crying","laughing","singing","cough","sneeze","knock","music","song",'sing']
                                
                                audio_keyword_count=0
                                for keyword in audio_keyword:

                                    if keyword in sent:
                                        audio_keyword_count=audio_keyword_count+1
                                        #temp=sent 
                                        #audio_annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})

 
                                if audio_keyword_count==0:
                                    if temp!=sent:
                                        annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})
                                        temp=sent
                                '''
                                if 'talk' in sent:
                                    continue
                                else:
                                    annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})
                                '''
            #with open('../../../hdd1/AST/AVSD_train_audio_top20keyword', 'w') as outfile:
                    #json.dump(data, outfile, indent=4)
            
            print("trainset length: ",len(annotations))
            #print("start < 2: ",train_count1)
            #print("duration-2 < end < duration: ",train_count2)
            #print("start < 2  and  duration-2 < end < duration: ",train_count3)



            self.annotations = annotations #

       
        elif self.split=='val'or'test':
            #print('test!!!!!!!!!!!!!')
            gt = json.load(open(self.data_dir+'/valid_set4DSTC10-AVSD+reason.json'))
            #gt = json.load(open(self.data_dir+'/train_set4DSTC8-AVSD+reason_test.json'))
            val_count1=0
            val_count2=0
            val_count3=0

            annotations = []
            audio_annotations=[]
            for d in gt['dialogs']:
                vid = d['image_id']
                for t, turn in enumerate(d['dialog'], 1):
                    #gts['%s_res%03d' % (vid, t)].append(self.filter(turn['answer']))
                    if 'reason' in turn:
                        for reason in turn['reason']:
                            #intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
                            #reason['timestamp'][0]=round(reason['timestamp'][0],4)
                            #reason['timestamp'][1]=round(reason['timestamp'][1],4)
                            sent=self.filter(turn['question'])+" [SEP] "+self.filter(turn['answer'])
                            #sent=self.filter(turn['answer'])

                            if reason['timestamp'][0]<reason['timestamp'][1]:
                                if reason['timestamp'][0]<2:
                                    val_count1=val_count1+1
                                if self.durations[vid]>=reason['timestamp'][1]>self.durations[vid]-2:
                                    val_count2=val_count2+1
                                if reason['timestamp'][0]<2 and self.durations[vid]>=reason['timestamp'][1]>self.durations[vid]-2:
                                    val_count3=val_count3+1
                                #annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})

                                #file=open('../../../hdd1/AST/audioset_keywordsDict/Action_actionSound.dict',"rb") 
                                #content=pickle.load(file)
                                #print(content)

                                #assert 0
                                audio_keyword=["audio","audible","noise","sound","hear anything",
                                "background sound","any noise","any audio","any sound","can you hear",
                                "do you hear","speak","talk","conversation","say anything","saying","dialogue",
                                "bark","meow","crying","laughing","singing","cough","sneeze","knock","music","song",'sing']
                                
                                audio_keyword_count=0

                                for keyword in audio_keyword:

                                    if keyword in sent:
                                        audio_keyword_count=audio_keyword_count+1
                                        #temp=sent 
                                        #audio_annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})

 
                                if audio_keyword_count==0:
                                    if temp1!=sent:
                                        annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})
                                        temp1=sent
                                                                                  
                                

            #print("testset length: ",len(annotations))
            #print("start < 2: ",val_count1)
            #print("duration-2 < end < duration: ",val_count2)
            #print("start < 2  and  duration-2 < end < duration: ",val_count3)

            self.audio_annotations=audio_annotations
            self.annotations = annotations #
            print("testset length: ",len(self.annotations))
            #assert 0
            #print("testset length _ audio: ",len(self.audio_annotations))
            #for i in range(len(self.audio_annotations)):
                #self.audio_query(i)

        


    def audio_query(self,index):
        description=self.audio_annotations[index]['description']
        video_id = self.audio_annotations[index]['video']
        duration = self.durations[video_id]
        print(video_id)
        print(description)
        print('gt timestamp:',self.annotations[index]['times'] )

        file=open('../../../hdd1/AST/audioset_class.dict',"rb") 
        content=pickle.load(file)
        #print(content)
        #print('----------------------------')

        index_dict = {j:i for i,j in content.items()}
        #print(index_dict)
        #print('----------------------------')


        prob_map = np.load(os.path.join('../../../hdd1/AST/validset_pad_ol24', f'{video_id}.npy'))
        #print(prob_map)
        row,column=prob_map.shape
        #print('----------------------------')
        audio_class=[]
        audio_class=np.argmax(prob_map, axis=1)
        #print(audio_class)
        print('----------------------------')

        for i in range(row):
            print(10*i,'~',10*(i+1))
            print(prob_map[i][audio_class[i]])
            if prob_map[i][audio_class[i]] <0.5:
                print('n')
            else:
                print(index_dict.get(audio_class[i]))
            print(' ')
        print('----------------------------')
        



        a=input()

        #print(description)
        #print(prob_map)

        #assert 0
        
        

 

    def __getitem__(self, index):
        #print(self.annotations[index])
        video_id = self.annotations[index]['video'] #video id
        gt_s_time, gt_e_time = self.annotations[index]['times'] #timestamp
        description = self.annotations[index]['description'] #query

        

        duration = self.durations[video_id] #총길이
        
        tokenized_text = self.tokenizer.tokenize(description)
        
        #tok=self.tokenizer(description,padding='max_length',max_length=100,truncation=True)

        #input_ids=tok["input_ids"]
        #attention_mask=tok["attention_mask"]
        #token_type_ids = tok["token_type_ids"]


        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()], dtype=torch.long) 

        #print(description) #ex) person they take a photograph from a box
        #print(word_idxs) #ex) tensor([ 899,   39,  190,    7, 7852,   25,    7, 1930])

        word_vectors = self.word_embedding(word_idxs) #torch.Size([8, 300])

        visual_input, visual_mask = self.get_video_features(video_id) #torch.Size([n_frames, 4096]), torch.Size([n_frames, 1])
        #audio_input, audio_mask=self.get_audio_features(video_id)
        # Time scaled to fixed size
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
        # visual_input = interpolate_to_fixed_length(visual_input)
        
        visual_input = average_to_fixed_length(visual_input) #torch.Size([256, 4096])
        #audio_input=average_to_fixed_length_audio(audio_input) #torch.Size([256, 128])

        
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
            'visual_input': visual_input, #visual_input, #torch.Size([256, 4096])
            
            #일단 visual 대신에 audio 넣어서 돌려보기 나중에 수정하기!!!!!!!!!
            
            'anno_idx': index, 
            'word_vectors': word_vectors, #torch.Size([n_words, 300])
            'txt_mask': torch.ones(word_vectors.shape[0], 1), #torch.Size([n_words, 1])
            #'token_type_ids': token_type_ids,
            'map_gt': torch.from_numpy(overlaps), #16*16
            'reg_gt': torch.tensor([gt_s_idx, gt_e_idx]),
            'duration': duration,
            'timestamp': self.annotations[index]['times'],
            'sent':description
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        #hdf5_file = h5py.File(os.path.join(self.data_dir, '{}_features.hdf5'.format(self.vis_input_type)), 'r') 
        #features = torch.from_numpy(hdf5_file[vid][:]).float() #해당하는 id의 비디오의 feature를 가져와서 numpy에서 torch로 변환 
        if self.split=='train':
            features = np.load(os.path.join(self.data_dir+'/features/video_feats', f'{vid}_rgb.npy'))
        else:
            features = np.load(os.path.join(self.data_dir+'/features/video_feats', f'{vid}_rgb.npy'))

        features = torch.from_numpy(features).float()
        #print('--features.shape: ',features.shape) 
        #torch.Size([vis_n_frames, 2048])

        if config.DATASET.NORMALIZE: 
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        #print('vis_mask.shape: ',vis_mask.shape) 
        #torch.Size([vis_n_frames, 2048])


        return features, vis_mask

    def get_audio_features(self,vid):
        if self.split=='train':
            stack_vggish = np.load(os.path.join(self.data_dir+'/features/vggish', f'{vid}.npy'))
        else: 
            stack_vggish = np.load(os.path.join(self.data_dir+'/features/vggish', f'{vid}.npy'))

        stack_vggish = torch.from_numpy(stack_vggish).float()
        #print('stack_vggish.shape: ',stack_vggish.shape) 
        #torch.Size([aud_n_frames, 2048])

        if config.DATASET.NORMALIZE:  #############
            stack_vggish = F.normalize(stack_vggish,dim=1)
        aud_mask = torch.ones((stack_vggish.shape[0], 1))
        #print('aud_mask.shape: ',aud_mask.shape) 
        #torch.Size([aud_n_frames, 1])

        
        return stack_vggish, aud_mask



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