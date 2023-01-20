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
from dataset_sign_clip import Sign_Isolated
from Conv3D import r2plus1d_18
from epoch_classes.train_epoch import train_epoch
from epoch_classes.validate_epoch import val_epoch
from collections import OrderedDict

import snoop

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    @snoop
    def forward(self, x, target, smoothing=0.1):
        print("target:", target.size())
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        #target_unsqueezed = target.unsqueeze(1)
        #nll_loss = -logprobs.gather(dim=-1, index=target_unsqueezed)
        #nll_loss = F.nll_loss(logprobs, target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



# Data and label paths
exp_name = 'rgb_final'
data_path = r"C:/Users/bencl/Desktop/SeniorFallSemester/EECS_581/ASLProject/Sign-Language-Translator/src/DataPreparation/val_frames/train_frames"
data_path2 = r"C:/Users/bencl/Desktop/SeniorFallSemester/EECS_581/ASLProject/Sign-Language-Translator/src/DataPreparation/val_frames/val_frames"
label_train_path = r"C:/Users/bencl/Desktop/SeniorFallSemester/EECS_581/ASLProject/Sign-Language-Translator/src/DataPreparation/val_frames/train_labels.csv"
label_val_path = r"C:/Users/bencl/Desktop/SeniorFallSemester/EECS_581/ASLProject/Sign-Language-Translator/src/DataPreparation/val_frames/test_labels.csv"
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
num_classes = 3
#epochs = 100
epochs = 3
#batch_size = 24
batch_size = 8
learning_rate = 1e-3#1e-3 Train 1e-4 Finetune
weight_decay = 1e-4 #1e-4
log_interval = 300
#sample_size = 128
sample_size = 240
sample_duration = 16
attention = False
drop_p = 0.0
hidden1, hidden2 = 512, 256

if __name__ == '__main__':
    #Train!
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = Sign_Isolated(data_path=data_path, label_path=label_train_path, frames=sample_duration,
        num_classes=num_classes, train=True, transform=transform)
    val_set = Sign_Isolated(data_path=data_path2, label_path=label_val_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)

    # Create model

    model = r2plus1d_18(pretrained=True, num_classes=500)
    # load pretrained
    checkpoint = torch.load('pretrained/slr_resnet2d+1.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    if phase == 'Train':
        model.fc1 = nn.Linear(model.fc1.in_features, num_classes)
    print(model)

    
    model = model.to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    # Start training
    if phase == 'Train':
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            print('lr: ', get_lr(optimizer))
            # Train the model
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

            # Validate the model
            val_loss = val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
            scheduler.step(val_loss)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_path, "sign_resnet2d+1_epoch{:03d}.pth".format(epoch+1)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
    elif phase == 'Test':
        logger.info("Testing Started".center(60, '#'))
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, phase=phase, exp_name=exp_name)

    logger.info("Finished".center(60, '#'))