# Analysis and Extensions of Adversarial Training for Video Classification

## Summary

This is the PyTorch code for our paper: Analysis and Extensions of Adversarial Training for Video Classification

This code includes training, fine-tuning and testing on different video datasets, including UCF-101, and HMDB-51.

## Reference

This code or pre-trained models are based on the following work:

[
Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,  
"Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",  
arXiv preprint, arXiv:2004.04968, 2020.
](https://arxiv.org/abs/2004.04968)

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4).  
All models are trained on Kinetics-700 (_K_), Moments in Time (_M_), STAIR-Actions (_S_), or merged datasets of them (_KM_, _KS_, _MS_, _KMS_).  
If you want to finetune the models on your dataset, you should specify the following options.

```misc
r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700
r3d18_KM_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 1039
r3d34_K_200ep.pth: --model resnet --model_depth 34 --n_pretrain_classes 700
r3d34_KM_200ep.pth: --model resnet --model_depth 34 --n_pretrain_classes 1039
r3d50_K_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 700
r3d50_KM_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1039
r3d50_KMS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1139
r3d50_KS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 800
r3d50_M_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 339
r3d50_MS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 439
r3d50_S_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 100
r3d101_K_200ep.pth: --model resnet --model_depth 101 --n_pretrain_classes 700
r3d101_KM_200ep.pth: --model resnet --model_depth 101 --n_pretrain_classes 1039
r3d152_K_200ep.pth: --model resnet --model_depth 152 --n_pretrain_classes 700
r3d152_KM_200ep.pth: --model resnet --model_depth 152 --n_pretrain_classes 1039
r3d200_K_200ep.pth: --model resnet --model_depth 200 --n_pretrain_classes 700
r3d200_KM_200ep.pth: --model resnet --model_depth 200 --n_pretrain_classes 1039
```

Old pretrained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
However, some modifcations are required to use the old pretrained models in the current scripts.

## Requirements

* [PyTorch](http://pytorch.org/) (ver. 0.4+ required)

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c soumith
```

* FFmpeg, FFprobe

* Python 3

## Preparation

### ActivityNet

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs mp4_video_dir_path jpg_video_dir_path activitynet
```

* Add fps infomartion into the json file ```util_scripts/add_fps_into_activitynet_json.py```

```bash
python -m util_scripts.add_fps_into_activitynet_json mp4_video_dir_path json_file_path
```

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs mp4_video_dir_path jpg_video_dir_path kinetics
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python -m util_scripts.kinetics_json csv_dir_path 700 jpg_video_dir_path jpg dst_json_path
```

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path hmdb51
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/hmdb51_json.py```
  * ```annotation_dir_path``` includes brush_hair_test_split1.txt, ...

```bash
python -m util_scripts.hmdb51_json annotation_dir_path jpg_video_dir_path dst_json_path
```

## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

Confirm all options.

```bash
python main.py -h
```

Train ResNets-50 on the Kinetics-700 dataset (700 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5 
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5
```

Calculate top-5 class probabilities of each video using a trained model (~/data/results/save_200.pth.)  
Note that ```inference_batch_size``` should be small because actual batch size is calculated by ```inference_batch_size * (n_video_frames / inference_stride)```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

Evaluate top-1 video accuracy of a recognition result (~/data/results/val.json).

```bash
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

Fine-tune fc layers of a pretrained model (~/data/models/resnet-50-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/resnet-50-kinetics.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

Adversarial Training with PGD Linf attacks 
```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5 \
--attack_type pgd_inf --eps 4 --attack_iter 10 --step_size 1
```

Adversarial Training with PGD Linf attacks 
```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5 \
--attack_type pgd_inf --eps 4 --attack_iter 10 --step_size 1
```

Adaptive Adversarial Training with PGD Linf attacks 
```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5 \
--attack_type pgd_inf --eps 12 --eps_range adaptive
```

Curriculum Adversarial Training with PGD Linf attacks 
```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5 \
--attack_type pgd_inf --eps 12 --eps_range curriculum
```

Generative Adversarial Training with PGD Linf attacks 
```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5 \
--attack_type pgd_inf --eps 4 --attack_iter 10 --step_size 1 --use_ape --ape_path ape/path
```
