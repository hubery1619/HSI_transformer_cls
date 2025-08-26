# Group-Aware-Hierarchical-Transformer

This repository is the official implementation for our IEEE TGRS 2022 paper:

[Hyperspectral image classification using group-aware hierarchical transformer](https://www.doi.org/10.1109/TGRS.2022.3207933)

Last update: September 20, 2022

## Requirements

python == 3.7.9, cuda == 11.1, and packages in `requirements.txt`

## Datasets

Download following datasets:

- [Houston (HU)](https://hyperspectral.ee.uh.edu/?page_id=459)
- [Botswana](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

- [Pavia University (PU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)


The three datasets are stored as follows:

```
datasets/
  hu/
    gt.mat
    HU_cube.mat
  bot/
    Botswana_gt.mat
    Botswana.mat
  pu/
    PaviaU_gt.mat
    PaviaU.mat
```

## Codes for Training and Validation

The model is trained based on the following command: 

```bash
python main.py --model proposed --dataset_name hrl --epoch 300 --bs 64 --device 0 --ratio 0.06
```

The model is evaluated based on the following command: 

```bash
python eval.py --model proposed --dataset_name hu --device 0 --trans_type 8 --patch_size 11 --weights ./checkpoints/proposed/hu/300/8/0.1/0
```


## Loss landscape analysis

```bash
python loss_landscape_analysis.py --model proposed --dataset_name hu --epoch 300 --bs 64 --ratio 0.1 --trans_type 0 --weights ./checkpoints/proposed/hu
```

## Acknowledgment
Our implementation is mainly based on the following codebases and paper. We gratefully thank the authors for their wonderful works.

1: https://github.com/amirgholami/PyHessian

2: https://github.com/MeiShaohui/Group-Aware-Hierarchical-Transformer


3: https://github.com/xxxnell/how-do-vits-work

