import os
import sys
from datetime import datetime #not sure we need this
import logging                  #not sure we need this
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms #might not need this
from Conv3D import flow_r2plus1d_18
from epoch_classes.train_epoch import train_epoch
from epoch_classes.validate_epoch import val_epoch
from collections import OrderedDict

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



# Data and label paths
exp_name = 'rgb_final'
data_path = "[PLACEHOLDER]/train_frames"
data_path2 = "[PLACEHOLDER]/val_frames"
label_train_path = "[PLACEHOLDER]/train_labels.csv"
label_val_path = "[PLACEHOLDER]/test_lables.csv"
model_path = "checkpoint/{}".format(exp_name)
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(os.path.join('results', exp_name)):
    os.mkdir(os.path.join('results', exp_name))

# log paths
log_path = "model_logs/logs/sign_resnet2d+1_{}_{:%Y-%m-%d_%H-%M-%S}.log".format(exp_name, datetime.now())
sum_path = "model_logs/runs/sign_resnet2d+1_{}_{:%Y-%m-%d_%H-%M-%S}".format(exp_name, datetime.now())
phase = 'Train'

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Used to select specifc GPUs - Pretty sure we do not need it
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# Use GPU if availible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 226 
epochs = 100
batch_size = 24
learning_rate = 1e-3#1e-3 Train 1e-4 Finetune
weight_decay = 1e-4 #1e-4
log_interval = 80
sample_size = 128
sample_duration = 32
attention = False
drop_p = 0.0
hidden1, hidden2 = 512, 256

if __name__ == '__main__':
    #Train!
    logger.info('Training...')