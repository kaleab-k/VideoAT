from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from mean import get_mean_std

from dataset import get_validation_data
from utils import Logger, worker_init_fn, get_lr
from validation import val_epoch

import pdb

from main import *
from utils import *

import logging
from typing import Union, Optional, Tuple

from art.classifiers import PyTorchClassifier
   
    
def make_model(
    model_status: str = "ucf101_trained", weights_path: Optional[str] = None
) -> Tuple[torch.nn.DataParallel, torch.optim.SGD]:
    statuses = ("ucf101_trained", "kinetics_pretrained")
    if model_status not in statuses:
        raise ValueError(f"model_status {model_status} not in {statuses}")
    trained = model_status == "ucf101_trained"
    if not trained and weights_path is None:
        raise ValueError("weights_path cannot be None for 'kinetics_pretrained'")

    opt = get_opt(['--attack_type','pgd_inf', '--resume_path', weights_path,
              '--n_classes', '101', '--model_depth', '101', '--model', 'resnext', 
              '--no_mean_norm', '--no_std_norm', '--ape_path', 'checkpoint/21.tar']) # '--use_ape'

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')


    model = generate_model(opt)

    if opt.use_ape:
        in_ch = 3
        G = Generator(in_ch).to(opt.device)

    model = resume_model(opt.resume_path, opt.arch, model)

    model = make_data_parallel(model, opt.distributed, opt.device)

    if opt.use_ape:
    #         G = make_data_parallel(G, opt.distributed, opt.device)
        if opt.ape_path is not None:
            checkpoint = torch.load(opt.ape_path)
    #             G.load_state_dict(checkpoint['generator'])
            G.load_state_dict(torch.load(opt.resume_path)['generator'])
            G = make_data_parallel(G, opt.distributed, opt.device)
        model = torch.nn.Sequential(G, model)

    logger.info(f"Loading model... {opt.model} {opt.model_depth}")

    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov,
    )

    return model, optimizer

def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model, optimizer = make_model(weights_path=weights_path, **model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(None, 240, 320, 3),
        channels_first=False,
        nb_classes=101,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model