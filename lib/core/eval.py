import json
import os
import argparse
import numpy as np
from terminaltables import AsciiTable
import sys
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from core.config import config, update_config
from train import logger


def set_iou( intervals_1, intervals_2):

    def within_interval(intervals, t):
        return sum([interval[0] <= t < interval[1] for interval in intervals]) > 0 
    tick = 0.1  # tick time [sec]
    intervals = intervals_1 + intervals_2 
    lower = int(min([intrv[0] for intrv in intervals]) / tick) 
    upper = int(max([intrv[1] for intrv in intervals]) / tick)

    intersection = 0
    union = 0
    for t in range(lower, upper):
        time = t * tick
        intersection += int(within_interval(intervals_1, time) & within_interval(intervals_2, time)) 
        union += int(within_interval(intervals_1, time) | within_interval(intervals_2, time)) 
    iou = float(intersection) / (union + 1e-8)
    return iou

def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list) 

    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)

    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]

    pred, gt = np.array(pred), np.array(gt)
    #print(gt) #ex) [[24.70000076 31.        ]]
    #print(gt[None,:,0]) #ex) [[24.70000076]]
    #print(gt[None,:,1]) #ex) [[31.]]
    #print(pred)
    #assert 0
    #print()
    #print()
    #assert 0

    inter_left = np.maximum(pred[:,0,None], gt[None,:,0]) #시작점들, gt의 시작점 중의 max
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1]) #끝점들, gt의 끝점 중의 min
    inter = np.maximum(0.0, inter_right - inter_left) #안겹치거나 s>e이면 0처리     #print()
    union_left = np.minimum(pred[:,0,None], gt[None,:,0]) #시작점들, gt의 시작점 중의 min
    union_right = np.maximum(pred[:,1,None], gt[None,:,1]) #끝점들, gt의 끝점 중의 max
    union = np.maximum(0.0, union_right - union_left) 

    union_array=np.array(union)
    if len(union_array[union_array==0]):
        print(gt)
        print(pred)
        print(union)
        assert 0

    try: 
        #overlap = 1.0 * inter / union #최종적으로 겹치는부분의 비율을 계산 
        overlap=1.0 * inter  / (union + 1e-8)
    except:
        print(inter)
        print(union)
        assert 0

    if not gt_is_list: 
        overlap = overlap[:,0]

    if not pred_is_list:
        overlap = overlap[0]

    return overlap

def rank(pred, gt):
    return pred.index(gt) + 1

def nms(dets, thresh=0.4, top_k=-1):
    #동일한 클래스에 대해 높은-낮은 confidence 순서로 정렬한다.
    #가장 confidence가 높은 boundingbox와 IOU가 일정 이상인 boundingbox는 동일한 물체를 detect했다고 판단하여 지운다.(16~20) 보통 50%(0.5)이상인 경우 지우는 경우를 종종 보았다


    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def eval(segments, data):

    #print(segments)


    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]
    #tious: 0.5,0.7 
    #recalls: 1,5

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    average_iou_2 = []
    test1=[]
    test2=[]
    test3=[]
    test4=[]

    gt = json.load(open('../../data/test_set4DSTC10-AVSD.json'))
         

    annotations = []
    audio_annotations=[] 
 
                    #intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
                    #reason['timestamp'][0]=round(reason['timestamp'][0],4)
                    #reason['timestamp'][1]=round(reason['timestamp'][1],4)
                #sent=self.filter(turn['question'])+" [SEP] "+self.filter(turn['answer'])
                #print(sent)
                    #sent=self.filter(turn['answer'])
                #annotations.append({'video':vid, 'times':reason['timestamp'], 'description': sent, 'duration': self.durations[vid]})



    

    for seg, dat in zip(segments, data):
        seg = nms(seg, thresh=config.TEST.NMS_THRESH, top_k=max_recall).tolist()

        #print('query: ', dat['description'])
        print('pd: ',seg[0])
        #[[3.467591, 6.9110994]]
        #[[5.6643146276474, 24.02862638235092]]
        #a=input()
        '''
        with open("./result.txt", mode="a") as file:
            file.writelines(['video: ',dat['video'],'\n'])
            file.writelines(['duration: ',str(dat['duration']),'\n'])
            file.writelines(['query: ', dat['description'],'\n'])
            file.writelines(['gt: ',str(dat['times']),'\n'])
            file.writelines(['pred: ',str(seg[0]),'\n'])
            file.write("----------------------------------------------------------") 
            file.write("\n")       
            file.write("\n")       
        '''
        '''
        if 'talk' in dat['description']:
            description=dat['description']
            video_id = dat['video']
            duration = str(dat['duration'])
            print(video_id)
            print(description)
            print('duration: ',duration)
            print('gt timestamp:',dat['times'])
            print('pred timestamp: ',seg[0])


            file=open('../../../hdd1/AST/audioset_class.dict',"rb") 
            content=pickle.load(file)
            #print(content)
            #print('----------------------------')

            index_dict = {j:i for i,j in content.items()}
            #print(index_dict)
            #print('----------------------------')

            threshold=0.4

            prob_map = np.load(os.path.join('../../../hdd1/AST/validset_pad_ol24', f'{video_id}.npy'))
            print(prob_map) # 확률값 매트릭스
     
            class_list=[0,1] #해당 쿼리에 관련있는 audio class의 인덱스 저장
            prob_map=prob_map[:,class_list] #prob_map에서 관련있는 audio class만 추출
            print(prob_map) 


            prob_map_sum=[]
            prob_map_sum=np.sum(prob_map,axis=0) #column 별로 확률값을 sum한 길이 #class의 리스트 
            print(prob_map_sum) 
            row,column=prob_map.shape



            audio_class=[]
            audio_max_sum=np.argmax(prob_map_sum) #가장 높은 sum을 가진 class 추출
            print('audio_max_sum:',audio_max_sum) #class index
            audio_class_sum=prob_map_sum[audio_max_sum]/row
            print('audio_class_sum:', audio_class_sum) #sum값을 row수로 나눈것
            audio_class_prob=[]
            audio_class_text=[]



            print('----------------------------')


            if audio_class_sum>=threshold:
                audio_class_text=index_dict.get(audio_max_sum)
                print(audio_class_text)

                print(prob_map[audio_max_sum][:])


                assert 0


                audio_class=np.argmax(prob_map, axis=1) #각 row마다 제일 높은 class 번호 저장
                print(audio_class)
                for i in range(row):
                    print(audio_class[i])
                    audio_class_prob.append(prob_map[i][audio_class[i]])


                print(audio_class_prob)

                #print('----------------------------')

                for i in range(row):
                    #print(10*i,'~',10*(i+1))
                    if audio_class_prob[i] <0.5:
                        audio_class_text.append('NONE')
                    else:
                        audio_class_text.append(index_dict.get(audio_class[i]))
                    #print(' ')
                #print('----------------------------')

                print(audio_class_text)
                assert 0

            '''


        test1.append(dat)
        test3.append( dat['times'])
        test4.append(seg[0])

        test2.append(dat['duration'])


        overlap = iou(seg, [dat['times']])
        overlap2=set_iou([seg[0]], [dat['times']])
        
        average_iou.append(np.mean(np.sort(overlap[0])[-3:])) #어짜피 이거는 top1만 포함
        
        average_iou_2.append(overlap2)
        #print(np.sort(overlap[0])[-3:])
        #print('average_iou: ',average_iou)
        #print('average_iou_2: ',average_iou_2)

        #print('----------------------------------------')
        #a=input()
        #print('overlap: ',overlap)
        #print('overlap2: ',overlap2)
        #assert 0

        for i,t in enumerate(tious): #tious: 0.5,0.7
            for j,r in enumerate(recalls): #recalls: 1,5
                eval_result[i][j].append((overlap > t)[:r].any())
    #print(test1[0:20])
    
    for i in range(1,15):
        str1="data:", "\t",test1[(i-1)*10:(i*10)-1]
        logger.info(str1)
        str2="duration:", "\t",test2[(i-1)*10:(i*10)-1]
        logger.info(str2)    
        str3="gt:", "\t",test3[(i-1)*10:(i*10)-1]
        logger.info(str3)      
        str4="pred:", "\t",test4[(i-1)*10:(i*10)-1]
        logger.info(str4)         
        logger.info("-----------------------------")       


    
    

        

    #print(average_iou[0:10])
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)
    #print(miou)
    miou2= np.mean(average_iou_2)
    #print(miou2)

    return eval_result, miou, miou2

def eval_predictions(segments, data, verbose=True):
    eval_result, miou, miou2 = eval(segments, data)
    #print('sssssssssss',miou2)

    if verbose:
        print(display_results(eval_result, miou, miou2, ''))

    return eval_result, miou, miou2

def display_results(eval_result, miou, miou2, title=None):
    #print('sssssssssss',miou2)
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']+['mIoU2']]
    eval_result = eval_result*100
    miou = miou*100
    miou2 = miou2*100
    #print(miou)
    #print(miou2)


    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        +['{:.02f}'.format(miou)]+['{:.02f}'.format(miou2)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose

if __name__ == '__main__':
    print('ssssssssssss')
    assert 0
    args = parse_args()
    reset_config(config, args)
    train_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/train_data.json', 'r'))
    val_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/val_data.json', 'r'))

    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prior = [list(item) for item in prior]
    prediction = [prior for d in val_data]

    eval_predictions(prediction, val_data)