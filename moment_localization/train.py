from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from logging import Logger

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math




# GPU 할당 변경하기
GPU_NUM = 1 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

print ('Current cuda device ', torch.cuda.current_device()) # check

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag

global logger
args = parse_args()
reset_config(config, args)
    # config 관련 설정


logger, final_output_dir = create_logger(config, args.cfg, config.TAG)


if __name__ == '__main__': #인터프리터로 직접 실행 하면 true

    

    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))
    # logger 관련 설정

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    # cudnn 관련 설정

    dataset_name = config.DATASET.NAME #Charades
    model_name = config.MODEL.NAME #TAN

    train_dataset = getattr(datasets, dataset_name)('train') # datasets.charades(train)
    if config.TEST.EVAL_TRAIN: 
        eval_train_dataset = getattr(datasets, dataset_name)('train') # datasets.charades(train)
    if not config.DATASET.NO_VAL: 
        val_dataset = getattr(datasets, dataset_name)('val') # datasets.charades(val)
    test_dataset = getattr(datasets, dataset_name)('test') # datasets.charades(test)


    model = getattr(models, model_name)() # models.tan
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE: #저장해놓은 체크포인트에서 시작할지
        print('------------------')
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint,strict=False)
    '''
        for param in model.parameters():
            param.requires_grad = False
        
        model.nn = nn.Linear(2048, 4096) 
    '''
    '''
    if torch.cuda.device_count() > 1: # GPU여러개 쓸때
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    '''
    
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    #optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(20,40), gamma=0.1)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=5, verbose=1)

    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset, 
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS, 
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        else:
            raise NotImplementedError

        return dataloader


    def network(sample):
        anno_idxs = sample['batch_anno_idxs']

        #input_ids = sample['batch_input_ids'].cuda()
        #attention_mask = sample['batch_attention_mask'].cuda()
        #token_type_ids = sample['batch_token_type_ids'].cuda()
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        #print(textual_mask)
        #a=input()
        visual_input = sample['batch_vis_input'].cuda()
        #print(visual_input)
        #a=input()
        map_gt = sample['batch_map_gt'].cuda()
        #print(map_gt)
        #a=input()
        duration = sample['batch_duration']
        #print(duration)
        #a=input()
        '''
        linear=nn.Linear(2048,4096).to(device)
        visual_input=linear(visual_input)
        '''
        prediction, map_mask = model(textual_input, textual_mask,  visual_input)
        #print(prediction)
        #a=input()
        #print(map_mask)
        #a=input()
        loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        return loss_value, sorted_times

    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

            sorted_indexs[:,1] = sorted_indexs[:,1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times

    def on_start(state): #1
        #print('----------on_start----------')
        state['loss_meter'] = AverageMeter() #Computes and stores the average and current value
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL) #12404/32*1
        print(state['test_interval'])
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval']) #####

    def on_forward(state): #2
        #print('----------on_forward----------')
        #`torch.nn.utils.clip_grad_norm_(model.parameters(), 10) #gradient가 일정 threshold를 넘어가면 clipping 
        #(gradient의 L2norm(norm이지만 보통 L2 norm사용)으로 나눠주는 방식)
        # 두번째파라미터는 max_norm
        state['loss_meter'].update(state['loss'].item(), 1) #####

    def on_update(state):# Save All #3
        #print('----------on_update----------')

        if config.VERBOSE:
            state['progress_bar'].update(1) #####

        if state['t'] % state['test_interval'] == 0: #training 한 에폭이 끝나면 training set과 testing set 두개 모두에서 성능평가 진행 #state['test_interval']
            #print("///////////////////////")
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg) #돌아간 배치수?, train loss
            table_message = ''
            
            if config.TEST.EVAL_TRAIN: #true
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],train_state['miou2'],
                                                   'performance on training set')
                table_message += '\n'+ train_table
            
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                #state['scheduler'].step(val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],val_state['miou2'],
                                                 'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],test_state['miou2'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message+table_message+'\n'
            logger.info(message)
            

            saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                dataset_name, model_name+'_'+config.DATASET.VIS_INPUT_TYPE,
                state['t'],  test_state['miou'], test_state['miou2']))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)
            
            #if torch.cuda.device_count() > 1:
                #torch.save(model.module.state_dict(), saved_model_filename)
            #else:
            
                torch.save(model.state_dict(), saved_model_filename)

            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()

    def save_at_logger(string1):
        logger.info(string1)

    def on_end(state):
        #print('----------on_end----------')
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        #print('----------on_test_start----------')
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        #print('----------on_test_foward----------')
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        '''
        print('=-======================================')

        print(min_idx)
        print()
        #batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        '''
        batch_indexs = range(len(state['output']))
        '''
        print( state['sample']['batch_duration'])
        print()
        print( state['sample']['batch_anno_idxs'])
        print()
        print(batch_indexs)
        print()
        print(len(state['output']))
        print()
        '''
        sorted_segments = [state['output'][i] for i in batch_indexs]
        #print(sorted_segments)
        #assert 0
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        #print('----------on_test_end----------')
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'],state['miou2'] = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)