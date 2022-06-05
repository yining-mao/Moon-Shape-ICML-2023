# moonshape_subpopulation

- code for paper *On the nonlinear correlation of ML performance between data subpopulations*

## Requirements
- mxnet >= 1.7.0
- torch >= 1.10.1
- torchvision >= 0.11.2
- autogluon
- gluoncv

## Datasets
We implement 5 subpopulation shift datasets with 6 settings (2 versions for Modified-CIFAR4). To prepare the data, run the jupyter notebook file in `datasets/` corresponding to each dataset. 
- For Metashift, PACS, OfficeHome, data needs to be downloaded;
- For Waterbirds, install WILDS using pip: `pip install wilds`;
- For Modified-CIFAR4, CIFAR10 dataset will be downloaded first with torchvision.
