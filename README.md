# Hyperspectral imagery classification investigation

This repository provides the implementation of the following work:

[Investigation of hierarchical spectral vision transformer architecture for classification of hyperspectral imagery (IEEE TGRS 2024)]

## 1. Requirements

```bash
conda env create -f environment.yml
```
Activate the conda environment as follows:
```bash
conda activate HSI_cls
```

## 2. Datasets sources

Download following datasets:

- [Houston (HU)](https://hyperspectral.ee.uh.edu/?page_id=459)
- [Botswana](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

- [Pavia University (PU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)


The datasets folder is organized as follows:

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

## 3. Codes for training and evaluation

To train the model, run the following command:

```bash
python main.py \
  --model <model_name> \
  --dataset_name <dataset> \
  --epoch <num_epochs> \
  --bs <batch_size> \
  --device <gpu_id> \
  --ratio <train_ratio>
  ```

To evaluate the model, run the following command:

```bash
python eval.py \
  --model <MODEL_NAME> \
  --dataset_name <DATASET_NAME> \
  --device <GPU_ID> \
  --trans_type <TRANSFORMER_TYPE> \
  --patch_size <PATCH_SIZE> \
  --weights <PATH_TO_WEIGHTS>
  ```


## 4. Loss landscape analysis

```bash
python loss_landscape_analysis.py \
  --model <MODEL_NAME> \
  --dataset_name <DATASET_NAME> \
  --epoch <NUM_EPOCHS> \
  --bs <BATCH_SIZE> \
  --ratio <TRAIN_RATIO> \
  --trans_type <TRANSFORMER_TYPE> \
  --weights <WEIGHTS_PATH>
```

## 5. Hessian Eigenvalue analysis

Step 1: generate hessian csv file.
```bash
python hessian_analysis.py \
  --model <MODEL_NAME> \
  --dataset_name <DATASET_NAME> \
  --epoch <NUM_EPOCHS> \
  --bs <BATCH_SIZE> \
  --ratio <TRAIN_RATIO> \
  --trans_type <TRANSFORMER_TYPE> \
  --weights <WEIGHTS_PATH> \
  --patch_size <PATCH_SIZE>
```
Step 2: plot Hessian figure.
```bash
python kernel_density_estimation.py --hessian_csv_path <PATH_TO_HESSIAN_CSV>
```

## Acknowledgment
Our implementation is mainly based on the following codebases and paper. We gratefully thank the authors for their excellent work.

1: https://github.com/amirgholami/PyHessian

2: https://github.com/MeiShaohui/Group-Aware-Hierarchical-Transformer

3: https://github.com/xxxnell/how-do-vits-work

