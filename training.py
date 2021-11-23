import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import numpy as np

## ----------------------- ART imports ----------------------- ##
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch as ProjectedGradientDescent
)
from art.attacks.evasion.frame_saliency import FrameSaliencyAttack

from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import OverTheAirFlickeringPyTorch as Flickering

from armory.art_experimental.attacks.pgd_patch import PGDPatch
from armory.art_experimental.attacks.video_frame_border import FrameBorderPatch


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False,
                attack_type="clean",
                eps=4,
                step_size=1,
                attack_iter=5,
                use_ape=False,
                D=None):
    
    
    print('train at epoch {}'.format(epoch))
    
    end_to_end = True if D is not None else False
    eps_rvs = None
    step_rvs = None
    attack_type_rvs = None
    
    if isinstance(eps, (list, np.ndarray)): 
        eps_rvs = eps
    if isinstance(step_size, (list, np.ndarray)): 
        step_rvs = step_size
    if isinstance(attack_type, (list, np.ndarray)): 
        attack_type_rvs = attack_type
        
#     step_dict = {0:0, 1:0.325, 2:0.5, 3:0.7, 4:1, 5:1.3, 6:1.45, 7:1.75, 8:1.85, 9:1.9, 10:2, 11:2, 12:2.27}
    step_dict = {0:0, 1:0.325, 2:0.44, 3:0.7, 4:1, 5:1.3, 6:1.45, 7:1.75, 8:1.85, 9:1.9, 10:2, 11:2, 12:2.27}

#     iter_dict = {0:0, 1:8, 2:8, 3:7, 4:6, 5:5, 6:4, 7:4, 8:3, 9:3, 10:3, 11:2, 12:2, 15:2}
#     iter_dict = {0:0, 1:12, 2:11, 3:10, 4:10, 5:9, 6:8, 7:7, 8:6, 9:5, 10:5, 11:5, 12:4, 15:4} 0.1*optim
#     iter_dict = {0:0, 1:8, 2:7, 3:6, 4:6, 5:5, 6:5, 7:4, 8:4, 9:3, 10:3, 11:2, 12:1, 15:1} #HMDB51

        
    if use_ape:
        model[0].train() if end_to_end else model[0].eval()
        model[1].train()
        if end_to_end:
            D.train()
            lr = 0.0002#0.0002
            opt_G = torch.optim.Adam(model[0].parameters(), lr=lr, betas=(0.5, 0.999))
    #         opt_G = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
            loss_bce_G = torch.nn.BCELoss(reduction='sum').cuda(device)
            loss_mse = torch.nn.MSELoss().cuda(device)
            opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
            loss_bce = torch.nn.BCELoss().cuda(device)
        
    else:
        model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
              
        if eps_rvs is not None:
            eps = eps_rvs[i]
        if step_rvs is not None:
            step_size = step_rvs[i]
        if attack_type_rvs is not None:
            attack_type = attack_type_rvs[i]
            
        data_time.update(time.time() - end_time)
        inputs = inputs.to(device)
        targets = targets.to(device, non_blocking=True)
        
        if attack_type == "clean" or eps==0:
            outputs = model(inputs)
        else:
            if attack_type != "pgd_inf":
                with torch.cuda.device(0):
                    model_art = PyTorchClassifier(
                        model,
                        loss=criterion,
                        optimizer=optimizer,
                        input_shape=(None, 240, 320, 3),
                        nb_classes=101,
                        clip_values=(0.0, 1.0),
                        device_type="gpu",
                    )
            if attack_type == "pgd_inf":
    #             attack_iter = np.random.randint(low=1, high=attack_iter+1)
#                 step_size = step_dict[int(eps)] #optimal step-size
                step_size = eps #0.1*(eps+1)
#                 attack_iter = iter_dict[int(eps)]
                adversary = LinfPGDAttack(predict=model, loss_fn=criterion,
                                             eps=float(eps/255), nb_iter=attack_iter, eps_iter=float(step_size/255))
                if use_ape and not end_to_end:
                    inputs = model[0](inputs)
                adv_inputs = adversary.perturb(inputs, targets)
#                 adversary = ProjectedGradientDescent(estimator=model_art, eps=float(eps/255), max_iter=attack_iter,
#                                                  eps_step=float(step_size/255), targeted=False, batch_size=len(inputs))
#                 adv_inputs = adversary.generate(inputs.cpu().numpy(), targets.cpu().numpy())
#                 adv_inputs = torch.from_numpy(adv_inputs).to(device)
            elif attack_type == "frame_saliency":
                attacker = ProjectedGradientDescent(estimator=model_art, eps=0.015, 
                                                         max_iter=100, eps_step=0.003, targeted=False, batch_size=len(inputs))#
                adversary = FrameSaliencyAttack(classifier=model_art, attacker=attacker, batch_size=len(inputs), method="one_shot")
                adv_inputs = adversary.generate(inputs.cpu().numpy())
                adv_inputs = torch.from_numpy(adv_inputs).to(device)
            elif attack_type == "frame_saliency_iter":
                attacker = ProjectedGradientDescent(estimator=model_art, eps=0.015, 
                                                         max_iter=100, eps_step=0.0004, targeted=False, batch_size=len(inputs))#
                adversary = FrameSaliencyAttack(classifier=model_art, attacker=attacker, batch_size=len(inputs), method="iterative_saliency")
                adv_inputs = adversary.generate(inputs.cpu().numpy())
                adv_inputs = torch.from_numpy(adv_inputs).to(device)

            elif attack_type == "patch_pgd":
                adversary = PGDPatch(estimator=model_art, eps=1.0, eps_step=0.005, max_iter=300, num_random_init=0, random_eps=False,  batch_size=len(inputs))
                adv_inputs = adversary.generate(inputs.cpu().numpy(), targets.cpu().numpy(), patch_ratio=0.15, video_input=True, xmin=0, ymin=0) 
                adv_inputs = torch.from_numpy(adv_inputs).to(device) 
            elif attack_type == "flickering":
                adversary = Flickering(classifier=model_art, eps_step=0.02, beta_0=2.0, beta_1=0.1, beta_2=0.9, max_iter=10)
                adv_inputs = adversary.generate(inputs.cpu().numpy())
                adv_inputs = torch.from_numpy(adv_inputs).to(device)
            elif attack_type == "frame_border":
                adversary = FrameBorderPatch(estimator=model_art, eps=1.0, eps_step=0.005, max_iter=300, batch_size=len(inputs))
                adv_inputs = adversary.generate(inputs.cpu().numpy(), targets.cpu().numpy(), patch_ratio=0.15)
                adv_inputs = torch.from_numpy(adv_inputs).to(device) 
                
            if use_ape and end_to_end: 
                eps1, eps2 = 0.7, 0.3
                current_size = adv_inputs.size(0)
                # Train D
                t_real = torch.autograd.Variable(torch.ones(current_size).to(device))#.to(f'cuda:{model.device_ids[0]}'))
                t_fake = torch.autograd.Variable(torch.zeros(current_size).to(device))#.to(f'cuda:{model.device_ids[0]}'))

                y_real = D(adv_inputs).squeeze()
                inputs_fake = model[0](adv_inputs)
                
                y_fake = D(inputs_fake).squeeze()
                loss_D = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                
                # Train G
                for _ in range(2):
                    
                    inputs_fake = model[0](adv_inputs)
                    y_fake = D(inputs_fake).squeeze()
                    ## L1, SSIM
                    loss_G = eps1 * loss_mse(inputs_fake, inputs) + eps2 * loss_bce(y_fake, t_real) 
                    opt_G.zero_grad()
                    loss_G.backward()
                    opt_G.step()
#                     outputs = model(adv_inputs)
#                     loss = criterion(outputs, targets)
#                     loss += loss_G
#                     opt_G.zero_grad()
#                     loss.backward()
#                     opt_G.step()
#             else:
            outputs = model(adv_inputs)
        
        loss = criterion(outputs, targets) 
#         L1_reg = torch.tensor(0., requires_grad=True)
#         for name, param in model.named_parameters():
#             if 'weight' in name:
#                 L1_reg = L1_reg + torch.norm(param, 1)

#         loss = loss + 1e-4 * L1_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#             outputs = model(adv_inputs)
        
#         loss = criterion(outputs, targets)
            
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if use_ape:
#             loss_G = loss_bce_G(x_fake, inputs)
#             opt_G.zero_grad()
#             loss_G.backward()
#             opt_G.step()


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
