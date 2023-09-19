Transformer Based Weakly Supervised Framework for Multi-Scaled Object Detection
========

PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer)-based WSOD(**W**eakly **S**upervised **O**bject **D**etection) framework.
We combined DETR with MIL-based WSOD framework to avoid expensive hand-crafted annotations. 

<img src="https://github.com/gkstlsgp3as/DETR_WSOD/assets/58411517/2108a55b-737c-4a1d-8e3a-cab25c7f36cf" width="700" height="450" align="center"/>
<img src="https://github.com/gkstlsgp3as/DETR_WSOD/assets/58411517/13339f8c-5476-4379-844e-bdb427f30d42" width="700" height="300" align="center"/>


**What it is**. 
state-of-the-art deep neural networks still suffer from detecting multi-scaled objects, especially small ones. To mitigate this problem, our study propose attention and similarity based pseudo bounding box generator to collectively detect multi-scaled objects, leading the model to employ class-agnostic representations of objects. Armed with this novel approach, Transformer based DETR architecture deploys multi-instance head and refinement head. 

For details see [Transformer Based Weakly Supervised Framework for Multi-Scaled Object Detection.pdf](https://github.com/gkstlsgp3as/DETR_WSOD/files/12116139/Transformer.Based.Weakly.Supervised.Framework.for.Multi-Scaled.Object.Detection.pdf) by Shinhye Han, Jeongwoo Shin, and Keunyoung Kim.

## Contents

1. [Basic installation](#basic-installation)
2. [Data preparation](#data-preparation)
3. [Training](#training)
4. [Codes structure](#codes)

# Basic Installation
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/gkstlsgp3as/DETR_WSOD.git
```
Then, install the packages required:
```
conda install -c pytorch pytorch torchvision
pip install -r requirements.txt
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.


## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  images/
    train2017/    # train images
    val2017/      # val images
    test2017/     # test images
```

## Training
You can train it on a single node with 4 gpus for 300 epochs with:

```shell
# for distributed training (resnet50 + wsod)
python -m torch.distributed.launch --nproc_per_node=4 main.py --coco_path /path/to/data/coco --backbone resnet50 --wsod --no_aux_loss --output_dir ./outputs/resnet50_wsod

# for distributed training (dino + wsod)
python -m torch.distributed.launch --nproc_per_node=4 main.py --coco_path /path/to/data/coco --backbone dino --arch vit_small --patch_size 16 --wsod --no_aux_loss --output_dir ./outputs/dino_wsod
```

For single-gpu training, you can simply use:
```
python main.py --coco_path /path/to/data/coco --backbone dino --arch vit_small --patch_size 16 --wsod 
```

Once you have a model checkpoint, you can resume training when it is accidently aborted :
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py--epochs 25 --lr_drop 15 --coco_path /path/to/coco  --resume ./outputs/dino_wsod/checkpoint.pth --no_aux_loss --wsod --backbone resnet50
```

A single epoch takes 1 hr, so 300 epoch training
takes around 12 days on a single machine with 4 GeForce 3090 cards.
To ease reproduction of DETR results we share the link
[results and training logs](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)
for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50 from DETR official github (https://github.com/facebookresearch/detr.git).

They train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```
We share results for all DETR detection models from official DETR github 
[gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).
Note that numbers vary depending on batch size (number of images) per GPU.
Non-DC5 models were trained with batch size 2, and DC5 with 1,
so DC5 models show a significant drop in AP if evaluated with more
than 1 image per GPU.

## Multinode training
Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):
```
pip install submitit
```
Train baseline DETR-6-6 model on 4 nodes for 300 epochs:
```
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

## Codes
Below is the code structure

- **models**: codes for DETR and WSOD framework
- **outputs**
    - **resnet50**: outputs for the model based on resnet50
    - **resnet50_wsod**: outputs for the model based on resnet50 with wsod framework
    - **dino_wsod**: outputs for the model based on dino with wsod framework
- **tools**: files for proposal generation from dino, and sample codes showcased with jupyter notebook
- **util**: files for loss computation and configuration
- **data**
    - **coco**: COCO dataset
- **datasets**: files for preprocessing or evaluating with dataset


