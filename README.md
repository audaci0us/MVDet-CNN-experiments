My personal experiments on CNN behind "Multiview Detection with Feature Perspective Transformation" [[Website](https://hou-yz.github.io/publication/2020-eccv2020-mvdet)] [[arXiv](https://arxiv.org/abs/2007.07247)]

```
@inproceedings{hou2020multiview,
  title={Multiview Detection with Feature Perspective Transformation},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={ECCV},
  year={2020}
}
```



## Overview
This repo is for my practice and trials in learning Convolutional Neural Networks. The changes carried out are probably simple ones. The primary goal here is to get familiar with a code base that's a good implementation of a Neural Network architecture.

Here I'm focusing on MVDet model, which has been described and analyzed in the paper linked at the top. I got my understandings by reading through this paper (still learning from it) and also by going through the code that the paper's authors provided (this repo is a fork of theirs). 
With the intention of learning, I'm trying out different things here. Hoping to document enough of the learnings! We'll see!

 
## Content
- [Original setup](#original-setup)
- [Experiment 1](#experimen-1)
    * [Trials attempted in exp 1](#trials-attempted-in-exp-1)
    * [Changes carried out in exp 1](#changes-carried-out-in-exp-1)
- [Experiment 2](#experiment-2)
    * [Trials attempted in exp 2](#trials-attempted-in-exp-2)
    * [Changes carried out in exp 2](#changes-carried-out-in-exp-2)


## Original setup
The original architecture of MVDet is given below.
![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet")


Their code implementation (and also my modified ones) of MVDet uses CUDA as well as the following libraries
- python 3.7+
- pytorch 1.4+ & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- matlab & matlabengine (required for evaluation) (see this [link](/multiview_detector/evaluation/README.md) for detailed guide)



## Experiment 1
This experiment is only to get some basic familiarity with the code logics.

### Trials attempted in exp 1
1. Importantly, MVDet uses a Resnet18 architecture as its model-core. I've modified the code to use Resnet34 instead.
2. Some setup related changes have been carried out in order to run the code in my Windows desktop with one GPU instead of 2 GPUs as run by the paper's authors.
3. Also, minor changes are done to the main script where the parameters are explicitly hard-coded (this is for my convenience).

### Changes carried out in exp 1
1. "persp_trans_detector.py" script has a "PerspTransDetector" class. In its constructor (`__init__` method), Resnet34 is added as an additional option.
2. In the same script, the `__init__` method and the `forward` method, both establish the device in which the model stays and the data gets loaded onto.
   - The original implementation splits the model between two GPUs. I remove that portion and instead put the whole model into the single available GPU (in `__init__` method).
   - Then, the data before it gets put through the model (in the `forward` method), it is loaded into GPU. Since the original model is split, this happens multiple times for the dataset. At those instances, they are just loaded into the same GPU.
3. In the "main.py" script, the argparse library is used to apply the parameters. I've modified to use a simple class with properties instead. This is just for my convenience.




## Experiment 2
This one is a work in progress. Will be committing the changes soon. I've tried to modify the architecture to add an additional model in between the single-view results and the multi-view aggregation layers. These modifications will primarily be in "res_proj_variant". Hoping to complete them soon and commit them here. Fingers crossed!!
![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet")

### Trials attempted in exp 2
\*\*\* Work in progress \*\*\*

### Changes carried out in exp 2
\*\*\* Work in progress \*\*\*
