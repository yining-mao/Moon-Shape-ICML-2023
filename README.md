# moonshape_subpopulation

- code for paper *On the nonlinear correlation of ML performance between data subpopulations*

## Requirements
- mxnet >= 1.7.0
- torch >= 1.10.1
- torchvision >= 0.11.2
- autogluon
- gluoncv

## Datasets
We implement 5 subpopulation shift datasets with 6 settings (2 versions for Modified-CIFAR4). To see the dataset samples and prepare the data, run the jupyter notebook in corresponding dataset folder in `datasets/`.
- For Metashift [[GoogleDrive]](https://drive.google.com/file/d/1P2kvXa_erLVHBqL_0RDe5HLmpnA1rz2I/view?usp=sharing), PACS [[GoogleDrive]](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd), OfficeHome [[GoogleDrive]](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), data needs to be downloaded to corresponding dataset folders in `datasets/`;
- For Waterbirds, install WILDS using pip: `pip install wilds`;
- For Modified-CIFAR4, CIFAR10 dataset will be downloaded first with torchvision.

For example, if you want to prepare Metashift dataset, you can open `datasets/metashift/metashift_prepare.ipynb` and run the code. In each dataset preparation notebook, you can change the `ROOT_PATH` and `EXP_ROOT_PATH` in the first code cell. The prepared data will be saved in `EXP_ROOT_PATH/data` in [Pytorch Image Folder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) Format:
- Training data in `EXP_ROOT_PATH/data/train`
- Validation data in `EXP_ROOT_PATH/data/majority-val` and `EXP_ROOT_PATH/data/minority-val`

## Training Process
Following the search space of AutoGluon and train 500 different ML models with varying configurations.
Here for each dataset, we implement with e 5 model architectures, 5 learning rates, 5 batch sizes, and 4 training durations:
```
@ag.args( # 5 models * 5 lr * 5 batch_size * 4 epochs = 500 configurations
    model = ag.space.Categorical(
        'mobilenetv3_small', 
        'resnet18_v1b', 
        'resnet50_v1', 
        'mobilenetv3_large', 
        'resnet101_v2', 
        ),
    lr = ag.space.Categorical(0.01, 0.005, 0.001, 0.0005, 0.0001), 
    batch_size = ag.space.Categorical(8, 16, 32, 64, 128), 
    epochs = ag.space.Categorical(1, 5, 10, 25)
    )
```

Specify the experiment directory, and you can train the models.
For example, if you prepare and save the data in `experiments/metashift/data`, run:
```
python main.py --exp-dir experiments/metashift
```
and you will get the following results in `experiments/metashift/result`:
- A table with evaluation results of each configuration,
- A 'majority subpopulation accuracy vs. minority subpopulation accuracy' plot corresponding to the table.
