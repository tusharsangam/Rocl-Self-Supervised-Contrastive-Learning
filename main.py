#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data_loader
import torch
from loss import pairwise_similarity, NT_xent
from models.resnet import ResNet18,ResNet50
from models.projector import Projector
import torch.optim as optim
from torchlars import LARS
from scheduler import GradualWarmupScheduler
from attack_lib import FastGradientSignUntargeted,RepresentationAdv
import os
import time
import argparse
from collections import OrderedDict
#from utils import progress_bar, checkpoint


# In[ ]:


parser = argparse.ArgumentParser(description='PyTorch RoCL training')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--check_point_model', type=str, default="start_fresh")
parser.add_argument('--check_point_projector', type=str, default="start_fresh")
parser.add_argument('--ngpu', type=int, default=2)
args = parser.parse_args()
multi_gpu = True
ngpu = args.ngpu
start_epoch = 0

# In[ ]:
args.local_rank =% ngpu
torch.cuda.set_device(args.local_rank)
world_size = ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)


# In[2]:


train_sampler , train_loader, test_loader = data_loader.get_loader(batch_size=int(512/ngpu), local_rank=args.local_rank)


# In[3]:


model = ResNet18(num_classes=10, contrastive_learning=True)
projector = Projector(expansion=1)
model.cuda()
projector.cuda()

if args.check_point_model != "start_fresh":
    model_states = torch.load(args.check_point_model)
    new_state_dict = OrderedDict()
    for k, v in model_states['model'].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model_states = torch.load(args.check_point_projector)
    new_state_dict = OrderedDict()
    for k, v in model_states['model'].items():
        name = k[7:]
        new_state_dict[name] = v
    projector.load_state_dict(new_state_dict)
    torch.set_rng_state(model_states["rng_state"])
    start_epoch = model_states["epoch"]
    if args.local_rank % ngpu == 0:
        print("checkpoint loaded")
    
model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model       = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
projector   = torch.nn.parallel.DistributedDataParallel(
                projector,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)

if args.local_rank % ngpu == 0:
    print("DDP model ready")
# In[4]:


epsilon = float(8/255)
alpha = float(2/255)
max_iters = 7
loss_type="sim"
regularize_type = 'other'
lr = 0.1
weight_decay = 1e-6
epochs = args.epoch
lr_multiplier = 15.0
lamda = float(512)
random_start = True
advtrain_type = "Rep" #Rep/None
temperature = 0.5


# In[5]:


RepAttack = RepresentationAdv(model, projector, epsilon=epsilon, alpha=alpha, min_val=0.0, max_val=1.0, max_iters=max_iters, _type="linf", loss_type=loss_type, regularize = regularize_type)


# In[6]:


model_params = []
model_params += model.parameters()
model_params += projector.parameters()


# In[7]:


base_optimizer  = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
optimizer   = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
if(args.check_point_model != "start_fresh"):
     optimizer.load_state_dict(torch.load(args.check_point_projector)["optimizer_state"])

# In[8]:


scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)


# In[9]:


def checkpoint(model, acc, epoch, optimizer, save_name_add=''):
    # Save checkpoint.
    print('Saving..')
    state = {
        'epoch': epoch,
        'acc': acc,
        'model': model.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }

    save_name = './checkpoint/ckpt_'
    save_name += save_name_add

    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, save_name)


# In[10]:


def train(epoch):
    
    print('\nEpoch: %d' % epoch)

    model.train()
    projector.train()

    train_sampler.set_epoch(epoch)
    scheduler_warmup.step()

    total_loss = 0
    reg_simloss = 0
    reg_loss = 0

    for batch_idx, (ori, inputs_1, inputs_2, label) in enumerate(train_loader):
        ori, inputs_1, inputs_2 = ori.cuda(), inputs_1.cuda() ,inputs_2.cuda()

        
        attack_target = inputs_2

        
        advinputs, adv_loss = RepAttack.get_loss(original_images=inputs_1, target = attack_target, optimizer=optimizer, weight= lamda, random_start=random_start)
        reg_loss    += adv_loss.data

        if not (advtrain_type == 'None'):
            inputs = torch.cat((inputs_1, inputs_2, advinputs))
        else:
            inputs = torch.cat((inputs_1, inputs_2))
        
        outputs = projector(model(inputs))
        similarity, gathered_outputs = pairwise_similarity(outputs, temperature=temperature, multi_gpu=ngpu, adv_type = advtrain_type) 
        
        simloss  = NT_xent(similarity, advtrain_type)
        
        if not (advtrain_type=='None'):
            loss = simloss + adv_loss
        else:
            loss = simloss
        
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        reg_simloss += simloss.data
        
        optimizer.step()

        '''
        if (args.local_rank % ngpu == 0):
            if 'Rep' in advtrain_type:
                print(batch_idx, len(train_loader),
                             'Loss: %.3f | SimLoss: %.3f | Adv: %.2f'
                             % (total_loss / (batch_idx + 1), reg_simloss / (batch_idx + 1), reg_loss / (batch_idx + 1)))
            else:
                print(batch_idx, len(train_loader),
                         'Loss: %.3f | Adv: %.3f'
                         % (total_loss/(batch_idx+1), reg_simloss/(batch_idx+1)))
        '''
        
    return (total_loss/batch_idx, reg_simloss/batch_idx)


# In[11]:


def test(epoch, train_loss):
    model.eval()
    projector.eval()
    if args.local_rank % ngpu == 0 and epoch%10 == 0:
        checkpoint(model, train_loss, epoch, optimizer, save_name_add='_epoch_'+str(epoch))
        checkpoint(projector, train_loss, epoch, optimizer, save_name_add=('_projector_epoch_' + str(epoch)))


# In[12]:


start_time = time.time()

if (args.local_rank % ngpu == 0):
    print("Starting from Epoch {}".format(start_epoch))

for epoch in range(start_epoch, epochs):
    train_loss, reg_loss = train(epoch)
    test(epoch, train_loss)
end_time = time.time()
print("Time taken for {} epoch {}".format(epochs, (end_time - start_time) ))


# In[ ]:




