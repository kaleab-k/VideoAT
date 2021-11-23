from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr, AverageMeter
from training import train_epoch
from validation import val_epoch
import inference

from advertorch.attacks import LinfPGDAttack

## ART
from art.estimators.classification.pytorch import PyTorchClassifier

## ----------------------- APE-GAN ----------------------- ##
from gan_models import GeneratorUCF3D as Generator, DiscriminatorUCF3D as Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
## ---------------------------- // ----------------------- ##         
from scipy.stats import loguniform, beta
from matplotlib import pyplot as plt


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt(args=None):
    opt = parse_opts(args)

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]
    
    # Create the result path
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    assert arch == checkpoint['arch']
    
    if 'resnext-101-kinetics-hmdb51_split1.pth' in resume_path.name:
        print('hmdb51_split1')
        checkpoint['state_dict'] = {str(key).replace("module.", "") : value for key, value in checkpoint['state_dict'].items()}
        
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    
    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt.video_path, opt.annotation_path,
                                   opt.dataset, opt.input_type, opt.file_type,
                                   spatial_transform, temporal_transform)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov
                    )

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones, gamma=0.1) #, gamma=0.1

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform,
                                               temporal_transform)
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    inference_data, collate_fn = get_inference_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, spatial_transform,
        temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler, use_ape=False, D=None):
    
    if use_ape:
        APE = model[0]
        model = model[1]
        
        if hasattr(APE, 'module'):
            ape_state_dict = APE.module.state_dict()
        else:
            ape_state_dict = APE.state_dict()
        if D is not None:  
            if hasattr(D, 'module'):
                D_state_dict = D.module.state_dict()
            else:
                D_state_dict = D.state_dict()
        
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'generator': ape_state_dict if use_ape else None,
        'discriminator': D_state_dict if D is not None else None,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    D = None
    ## 3D-APE-GAN
    if opt.use_ape:
        in_ch = 3
        G = Generator(in_ch).to(opt.device)
#         D = Discriminator(in_ch).to(opt.device)
    
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
           
    model = make_data_parallel(model, opt.distributed, opt.device)
    
    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()
    
    ## 3D-APE-GAN
    if opt.use_ape:
        G = make_data_parallel(G, opt.distributed, opt.device)
#         D = make_data_parallel(D, opt.distributed, opt.device) # end-to-end
        if opt.ape_path is not None:
            checkpoint = torch.load(os.path.join(opt.ape_path))
            G.load_state_dict(checkpoint['generator'])
#             D.load_state_dict(checkpoint['discriminator']) # end-to-end
            G.eval()
        model = torch.nn.Sequential(G, model)
    
    # end-to-end G+C
#     parameters = model.parameters()
    
    if opt.is_master_node:
        print(model)
    
    criterion = CrossEntropyLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    
#     eps_rvs = opt.eps - (loguniform.rvs(0.5, opt.eps, size=opt.n_epochs + 1))
#     step_rvs_pre = np.linspace(0, opt.step_size, int(0.3*opt.n_epochs)) 
#     step_rvs = np.linspace(opt.step_size-0.2, opt.step_size+0.2, int(0.8*opt.n_epochs))
#     step_rvs = np.append(step_rvs_pre, step_rvs)
    if opt.eps_range == '':
        opt.eps_range = False
    elif (opt.eps_range == 'curriculum'):
        eps_rvs = np.linspace(1, opt.eps, opt.n_epochs+1)
#     eps_rvs = np.linspace(opt.eps, 1, opt.n_epochs+1)
    elif (opt.eps_range == 'log'):
        eps_rvs = opt.eps - (loguniform.rvs(0.5, opt.eps, size=opt.n_epochs + 1))
        eps_rvs = np.sort(eps_rvs)
    elif (opt.eps_range == 'beta'):
        a, b = 1.5,1.5
        eps_rvs = beta.rvs(a, b, size=opt.n_epochs + 1)*12
#     elif (opt.eps_range == 'multi_attack'):
        
#     iter_rvs = []
    step_dict = {0:0, 1:0.325, 2:0.5, 3:0.7, 4:1, 5:1.3, 6:1.45, 7:1.75, 8:1.85, 9:1.9, 10:2, 11:2, 12:2.27, 15:2.4} 
    iter_dict = {0:0, 1:8, 2:8, 3:7, 4:6, 5:5, 6:4, 7:4, 8:3, 9:3, 10:3, 11:2, 12:2, 15:2}

#     if opt.attack_type == "multi":
#         attack_opts = ["pgd_inf", "patch_pgd"]
#         multi_attack_list = np.random.randint(0, len(attack_opts), size=opt.n_epochs + 1)
                                          
    print('begin epoch:', opt.begin_epoch)
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            ''' 
            Adaptive eps values 
            '''
            epb = False # fixed epoch per batch
                
            if opt.eps_range == 'adaptive':
                print('auto eps distribution set')
                if i == opt.begin_epoch:
                    if opt.no_val:
                        val_loader, val_logger = get_val_utils(opt)
                    linf_attack_range = np.linspace(0, opt.eps, 10)
                    
                if i == opt.begin_epoch or (i % 10) == 0:
                    model.eval()
                    eps_loss = []
                    eps_prob = []
                    for eps in linf_attack_range:  
                        losses = AverageMeter()
                        for (inputs, targets) in (val_loader): 
                            inputs = inputs.to(opt.device)
                            targets = targets.to(opt.device, non_blocking=True)
                            step_size = step_dict[int(eps)]
                            adversary = LinfPGDAttack(predict=model, loss_fn=criterion,
                                                      eps=float(eps/255), nb_iter=opt.attack_iter, eps_iter=float(step_size/255))
#                             adversary.eps = float(eps/255)
                            adv_inputs = adversary.perturb(inputs, targets)
                            
                            with torch.no_grad():
                                outputs = model(adv_inputs)
                                loss = criterion(outputs, targets) 

                                losses.update(loss.item(), inputs.size(0))
#                             del adv_inputs; del outputs;
                        
                        eps_loss.append(losses.val)
                    eps_prob = [ x / sum(eps_loss) for x in eps_loss]
#                     print(i, ': ', sum(eps_loss))
                    model.train()
                    
                    eps_rvs = np.random.choice(linf_attack_range, p=eps_prob, size=len(train_loader)+1)
                    epb = False              
                    if not opt.no_val:
                        eps_rvs_val = np.random.choice(linf_attack_range, p=eps_prob, size=len(val_loader)+1)

                    plt.clf()
                    plt.hist(eps_rvs, density=True, histtype='stepfilled', alpha=1.0)
                    plt.title('epoch #: ' + str(i)) 
                    plt.savefig(os.path.join(opt.result_path, 'adaptive_' + '{0:03d}'.format(i) + '.png'))       
            # Set PGD parameters
#             eps = eps_rvs[i-1]
#             eps_val = eps_rvs[i-1]
            if opt.eps_range != False and epb:
                eps = eps_rvs[i-1]
                eps_val = eps_rvs[i-1]
                print('eps per batch', eps)
            elif opt.eps_range != False:
                print('eps_rvs per batch')
                eps = eps_rvs
                eps_val = eps_rvs_val if not opt.no_val else None
            else:
                eps = opt.eps
                eps_val = eps
            step_size = opt.step_size
            attack_iter = opt.attack_iter
            
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer)
            
            model_art=None
            
#             if opt.attack_type != "pgd_inf":
#                 with torch.cuda.device(0):
#                     model_art = PyTorchClassifier(
#                                 model,
#                                 loss=criterion,
#                                 optimizer=optimizer,
#                                 input_shape=(None, 240, 320, 3),
#                                 nb_classes=101,
#                                 clip_values=(0.0, 1.0),
#                                 device_type="gpu",
#                             )
                    
            if opt.attack_type == "multi":
#                 attack_type = attack_opts[multi_attack_list[i-1]]
                attack_opts = ["frame_saliency", "frame_saliency_iter", "patch_pgd", "frame_border"]
                multi_attack_list =  np.random.randint(0, len(attack_opts), size=len(train_loader)+1)
                attack_type = [attack_opts[attack_i] for attack_i in  multi_attack_list]
            else:
                attack_type = opt.attack_type
                
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed,
                        attack_type, eps, step_size, attack_iter, opt.use_ape, D)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler, opt.use_ape, D)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed, "pgd_inf", #opt.attack_type
                                      eps_val, step_size, attack_iter)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    if opt.inference:
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk, opt.attack_type, opt.eps, 
                            opt.step_size, opt.attack_iter)


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
