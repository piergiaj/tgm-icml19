from __future__ import division
import time
import os
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-model_file', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='1')
parser.add_argument('-dataset', type=str, default='charades')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np
import json

import models
from apmeter import APMeter

batch_size = 16
if args.dataset == 'multithumos':
    from multithumos_i3d_per_video import MultiThumos as Dataset
    from multithumos_i3d_per_video import mt_collate_fn as collate_fn
    train_split = 'multithumos.json'
    test_split = 'multithumos.json'
    rgb_root = '/ssd2/thumos/i3d_rgb'
    flow_root = '/ssd2/thumos/i3d_flow'
    classes = 65
elif args.dataset == 'charades':
    from charades_i3d_per_video import MultiThumos as Dataset
    from charades_i3d_per_video import mt_collate_fn as collate_fn
    train_split = 'charades/charades.json'
    test_split = 'charades/charades.json'
    rgb_root = '/ssd2/charades/i3d_rgb'
    flow_root = '/ssd2/charades/i3d_flow'
    classes = 157
elif args.dataset == 'mlb':
    from mlb_i3d_per_video import MLB as Dataset
    from mlb_i3d_per_video import mlb_collate_fn as collate_fn
    train_split = 'mlb/mlb.json'
    test_split = train_split
    rgb_root = '/ssd2/mlb/i3d_rgb'
    flow_root = '/ssd2/mlb/i3d_flow'
    classes = 8


def sigmoid(x):
    return 1/(1+np.exp(-x))

def load_data(train_split, val_split, root):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:
        
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, num_epochs=50):
    since = time.time()

    best_loss = 10000
    for epoch in range(num_epochs):
        print 'Epoch {}/{}'.format(epoch, num_epochs - 1)
        print '-' * 10

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_step(model, gpu, optimizer, dataloader['train'])
            prob_val, val_loss = val_step(model, gpu, dataloader['val'])
            probs.append(prob_val)
            sched.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'models/'+model_file)

def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1]/other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results

def run_network(model, data, gpu, baseline=True):
    # get the inputs
    inputs, mask, labels, other = data
    
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))
    
    cls_wts = torch.FloatTensor([1.00]).cuda(gpu)

    # forward
    if not baseline:
        outputs = model([inputs, torch.sum(mask, 1)])
        outputs = outputs.permute(0,2,1)
    else:
        outputs = model(inputs)
        outputs = outputs.squeeze(3).squeeze(3).permute(0,2,1) # remove spatial dims
    probs = F.sigmoid(outputs) * mask.unsqueeze(2)
    
    # binary action-prediction loss
    loss = F.binary_cross_entropy_with_logits(outputs, labels, size_average=False)#, weight=cls_wts)

    
    loss = torch.sum(loss) / torch.sum(mask) # mean over valid entries
    
    # compute accuracy
    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs, loss, probs, corr/tot
            
                

def train_step(model, gpu, optimizer, dataloader):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    
    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        optimizer.zero_grad()
        
        outputs, loss, probs, err = run_network(model, data, gpu)
        
        error += err.data[0]
        tot_loss += loss.data[0]
        
        loss.backward()
        optimizer.step()
    optimizer.step()
    optimizer.zero_grad()
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    print 'train-{} Loss: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, error)

  

def val_step(model, gpu, dataloader):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}


    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]
        
        outputs, loss, probs, err = run_network(model, data, gpu)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        
        error += err.data[0]
        tot_loss += loss.data[0]
        
        # post-process preds
        outputs = outputs.squeeze()
        probs = probs.squeeze()
        fps = outputs.size()[1]/other[1][0]
        full_probs[other[0][0]] = (probs.data.cpu().numpy().T, fps)
        
        
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    print 'val-map:', apm.value().mean()
    apm.reset()
    print 'val-{} Loss: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, error)
    
    return full_probs, epoch_loss


model_f = models.get_hier2

if __name__ == '__main__':

    if args.mode == 'flow':
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)


    if args.train:
        model = nn.DataParallel(model_f(classes))
    
        lr = 0.1*batch_size/len(datasets['train'])
        print lr
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        
        run([(model,0,dataloaders,optimizer, lr_sched, args.model_file)], num_epochs=60)

    else:
        print 'Evaluating...'
        rgb_model = nn.DataParallel(torch.load(args.rgb_model_file)
        rgb_model.train(False)
        
        dataloaders, datasets = load_data('', test_split, flow_root)

        rgb_results = eval_model(rgb_model, dataloaders['val'])

        flow_model = nn.DataParallel(torch.load(args.flow_model_files)
        flow_model.train(False)
            
        dataloaders, datasets = load_data('', test_split, flow_root)
            
        flow_results = eval_model(flow_model, dataloaders['val'])

        apm = APMeter()


        for vid in rgb_results.keys():
            o,p,l,fps = rgb_results[vid]
         
            if vid in flow_results:
                o2,p2,l2,fps = flow_results[vid]
                o = (o[:o2.shape[0]]*.5+o2*.5)
                p = (p[:p2.shape[0]]*.5+p2*.5)
            apm.add(sigmoid(o), l)
        print 'MAP:', apm.value().mean()
