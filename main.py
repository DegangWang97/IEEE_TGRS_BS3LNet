"""
See more details in papers:
  [1] L. Gao, D. Wang, L. Zhuang, X. Sun, M. Huang, and A. Plaza, 
      “BS3LNet: A New Blind-Spot Self-Supervised Learning Network for 
      Hyperspectral Anomaly Detection,” IEEE Trans. Geosci. Remote Sens., 
      vol. 61, 2023, Art. no. 5504218. DOI: 10.1109/TGRS.2023.3246565
      URL: https://ieeexplore.ieee.org/abstract/document/10049187

-----------------------------------------------------------------------
Copyright (March, 2023):    
            Lianru Gao (gaolr@aircas.ac.cn)
            Degang Wang (wangdegang20@mails.ucas.ac.cn)
            Lina Zhuang (zhuangln@aircas.ac.cn)
            Xu Sun (sunxu@aircas.ac.cn)
            Min Huang (huangmin@aircas.ac.cn)
            Antonio Plaza (aplaza@unex.es)

BS3LNet is distributed under the terms of the GNU General Public License 2.0.

Permission to use, copy, modify, and distribute this software for
any purpose without fee is hereby granted, provided that this entire
notice is included in all copies of any software which is or includes
a copy or modification of this software and in all copies of the
supporting documentation for such software.
This software is being provided "as is", without any express or
implied warranty. In particular, the authors do not make any
representation or warranty of any kind concerning the merchantability
of this software or its fitness for any particular purpose.
---------------------------------------------------------------------
"""

import argparse

from model import BS3LNet
from dataset import BS3LNetData
from utils import get_auc, setup_seed, TensorToHSI, init_weights

import torch
from torch import optim
import torch.nn as nn
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time


class Trainer(object):
    '''
    Trains a model
    '''
    def __init__(self, 
                opt,
                model,
                criterion,
                optimizer,
                dataloader,
                device,
                model_path: str,
                logs_path: str,
                save_freq: int=50,
                scheduler = None):
        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training sets provided in `dataloader`
        using `criterion` and `optimizer`.
        Saves model weight snapshots every `save_freq` epochs and saves the
        weights at the end of training.
        Parameters
        ----------
        model : torch model object, with callable `forward` method.
        criterion : callable taking inputs and targets, returning loss.
        optimizer : torch.optim optimizer.
        dataloader : train dataloaders.
        model_path : string. output path for model.
        logs_path : string. output path for log.
        save_freq : integer. Number of epochs between model checkpoints. Default = 50.
        scheduler : learning rate scheduler.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.model_path = model_path
        self.logs_path = logs_path
        self.save_freq = save_freq
        self.scheduler = scheduler
        self.opt = opt
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
            
        self.log_output = open(f"{self.logs_path}/log.txt", 'w')
        
        self.writer = SummaryWriter(logs_path)
        
        print(self.opt)
        print(self.opt, file=self.log_output)
        
    def train_epoch(self) -> None:
        # Run a train phase for each epoch
        self.model.train(True)
        loss_train = []
        
        for i, data in enumerate(self.dataloader):
            
            label = data['label'].to(self.device)
            input = data['input'].to(self.device)
            mask = data['mask'].to(self.device)
            
            # forward net
            output = self.model(input)
            
            # backward net
            self.optimizer.zero_grad()
            
            loss = self.criterion(output * (1 - mask), label * (1 - mask))
            
            loss.backward()
            self.optimizer.step()
            
            # get losses
            loss_train += [loss.item()]
            
            print("iter: " + str(i)
                  + "\tTrain Loss:" + str(round(np.mean(loss_train), 4)))
            
            print("iter: " + str(i)
                  + "\tTrain Loss:" + str(round(np.mean(loss_train), 4)), file = self.log_output)
            
        # ============ TensorBoard logging ============#
        # Log the scalar values
        info = {
            'Loss_train': np.mean(loss_train)
            }
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.epoch + 1)
            
        # Saving model
        if ((self.epoch + 1) % self.save_freq == 0):
            torch.save(self.model.state_dict(), os.path.join(self.model_path, 'BS3LNet' + '_' + self.opt.dataset + '_' + str(self.epoch + 1) + '.pkl'))

    def train(self) -> nn.Module:
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs))
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs), file = self.log_output)
            print('-' * 50)
            # run training epoch
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
        return self.model
       
         
def train_model(opt):
    
    DB = opt.dataset
    
    expr_dir = os.path.join('./checkpoints/', DB)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    prefix = 'BS3LNet' + '_batch_size_' + str(opt.batch_size) + '_epoch_' + str(opt.epochs)+ '_learning_rate_' + str(opt.learning_rate) + \
        '_patch_' + str(opt.patch) + '_size_window_' + str(opt.size_window) + '_ratio_' + str(opt.ratio) + '_gpu_ids_' + str(opt.gpu_ids)
    
    trainfile = os.path.join(expr_dir, prefix)
    if not os.path.exists(trainfile):
        os.makedirs(trainfile)
    
    # Device
    device = torch.device('cuda:{}'.format(opt.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu')
    
    # Directories for storing model and loss
    model_path = os.path.join(trainfile, 'model')
    
    logs_path = os.path.join(trainfile, './logs')
    
    setup_seed(opt.seed)
    
    loader_train, band = BS3LNetData(opt)
    net = BS3LNet(band, band, nch_ker=opt.nch_ker, norm=opt.norm_mode, nblk=opt.nblk).to(device)
    
    # Initialize net parameters
    init_weights(net, init_type=opt.init_weight_type, init_gain=opt.init_gain)
    
    # Define Optimizers and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
    scheduler_net = None
    
    if opt.lossm.lower() == 'l1':
        criterion = nn.L1Loss().to(device)  # Regression loss: L1
    elif opt.lossm.lower() == 'l2':
        criterion = nn.MSELoss().to(device)  # Regression loss: L2
    
    if torch.cuda.is_available():
        print('Model moved to CUDA compute device.')
    else:
        print('No CUDA available, running on CPU!')
    
    # Training
    t_begin = time.time()
    trainer = Trainer(opt,
                      net,
                      criterion,
                      optimizer,
                      loader_train,
                      device,
                      model_path,
                      logs_path,
                      scheduler=scheduler_net)
    trainer.train()
    t_end = time.time()
    print('Time of training-{}s'.format((t_end - t_begin)))


def predict(opt):
    
    DB = opt.dataset
    
    expr_dir = os.path.join('./checkpoints/', DB)
    prefix = 'BS3LNet' + '_batch_size_' + str(opt.batch_size) + '_epoch_' + str(opt.epochs)+ '_learning_rate_' + str(opt.learning_rate) + \
        '_patch_' + str(opt.patch) + '_size_window_' + str(opt.size_window) + '_ratio_' + str(opt.ratio) + '_gpu_ids_' + str(opt.gpu_ids)
    
    trainfile = os.path.join(expr_dir, prefix)
    
    model_path = os.path.join(trainfile, 'model')
    
    expr_dirs = os.path.join('./result/', DB)
    if not os.path.exists(expr_dirs):
        os.makedirs(expr_dirs)
    
    log_output = open(f"{expr_dirs}/log.txt", 'w')
    
    model_weights = os.path.join(model_path, 'BS3LNet' + '_' + opt.dataset + '_' + str(opt.epochs) + '.pkl')
    
    # test datalodar
    data_dir = './data/'
    image_file = data_dir + opt.dataset + '.mat'
    
    input_data = sio.loadmat(image_file)
    image = input_data['data']
    image = image.astype(np.float32)
    gt = input_data['map']
    gt = gt.astype(np.float32)
    
    band = image.shape[2]
    
    test_data = np.expand_dims(image, axis=0)
    loader_test = torch.from_numpy(test_data.transpose(0,3,1,2)).type(torch.FloatTensor)
    
    # Device
    device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    
    net = BS3LNet(band, band, nch_ker=opt.nch_ker, norm=opt.norm_mode, nblk=opt.nblk).to(device)
    net.load_state_dict(torch.load(model_weights, map_location = 'cuda:0'))
    
    t_begin = time.time()
    
    net.eval()
    test_data = loader_test
    img_old = test_data.to(device)
    
    img_new = net(img_old)
    
    HSI_old = TensorToHSI(img_old)
    HSI_new = TensorToHSI(img_new)
    
    auc, detectmap = get_auc(HSI_old, HSI_new, gt)
    
    t_end = time.time()
    
    print("AUC: " + str(auc))
    print("AUC: " + str(auc), file = log_output)
    
    print('Time of testing-{}s'.format((t_end - t_begin)))
    print('Time of testing-{}s'.format((t_end - t_begin)), file = log_output)
    
    sio.savemat(os.path.join(expr_dirs, 'detectmap.mat'), {'detectmap':detectmap})
    sio.savemat(os.path.join(expr_dirs, 'reconstructed_data.mat'), {'reconstructed_data':HSI_new})
    
def main():    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--gpu_ids", default=0, type=int, help='gpu ids: e.g. 0 1 2')
    
    parser.add_argument('--command', default='train', type=str, help='action to perform. {train, predict}.')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    
    parser.add_argument('--patch', default=19, type=int, help='input patch size')
    parser.add_argument('--size_window', default=5, type=int, help='candidate window size')
    parser.add_argument('--ratio', default=0.9, type=float, help='candidate pixel ratio (1-ratio)')
    
    parser.add_argument('--nch_ker', default=64, type=int, help='number of nch_ker')
    parser.add_argument('--nblk', default=2, type=int, help='number of nblk')
    parser.add_argument('--norm_mode', choices=['bnorm','inorm'], default='bnorm', help='norm_mode to use')
    parser.add_argument('--init_weight_type', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], default='normal', help='the name of an initialization method: normal | xavier | kaiming | orthogonal')
    parser.add_argument('--init_gain', default=0.02, type=float, help='scaling factor for normal, xavier and orthogonal')
    
    parser.add_argument('--lossm', default='l1', type=str, help='loss function for model training. one of ["l1", "l2"].')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='network parameter regularization')
    
    parser.add_argument('--epochs', default=1000, type=int, help='number of epoch')
    parser.add_argument('--dataset', default='AVIRIS-II', type=str, help='dataset to use')
    
    opt = parser.parse_args()
    
    if opt.command.lower() == 'train':
        train_model(opt)
    elif opt.command.lower() == 'predict':
        predict(opt)
    return
    
if __name__ == '__main__':
    
    main()