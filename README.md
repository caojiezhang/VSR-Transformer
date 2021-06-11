# VSR-Transformer


If you use this code for a paper please cite:

```
@article{cao2021vsrt,
  title={Video Super-Resolution Transformer},
  author={Cao, Jiezhang and Li, Yawei and Zhang, Kai and Van Gool, Luc},
  journal={arXiv},
  year={2021}
}
```

This repository is implemented based on [BasicSR](https://github.com/xinntao/BasicSR). If you use the repository, please consider citing BasicSR.

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repository

    ```bash
    git clone https://github.com/caojiezhang/VSR-Transformer.git
    ```

2. Install dependent packages

    ```bash
    cd VSR-Transformer
    pip install -r requirements.txt
    ```

3. Compile environment

    ```bash
    python setup.py develop
    ```

    You may also want to specify the CUDA paths:

    ```bash
    CUDA_HOME=/usr/local/cuda \
    CUDNN_INCLUDE_DIR=/usr/local/cuda \
    CUDNN_LIB_DIR=/usr/local/cuda \
    python setup.py develop
    ```

## Dataset Preparation

- Please refer to **[DatasetPreparation.md](docs/DatasetPreparation.md)** for more details.
- The descriptions of currently supported datasets (`torch.utils.data.Dataset` classes) are in [Datasets.md](docs/Datasets.md).


## Train and Test

- **Training and testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.
- **Options/Configs**: Please refer to [Config.md](docs/Config.md).
- **Logging**: Please refer to [Logging.md](docs/Logging.md).


    ```bash
    # Train on REDS
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/train_vsrTransformer_x4_REDS.yml --launcher pytorch

    # Train on Vimeo-90K
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/train_vsrTransformer_x4_Vimeo.yml --launcher pytorch
    ```


### Test on REDS

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_vsrTransformer_x4_REDS.yml --launcher pytorch
    ```

### Test on Vimeo-90K

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_vsrTransformer_x4_Vimeo.yml --launcher pytorch
    ```

### Test on Vid4

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_vsrTransformer_x4_Vid4.yml --launcher pytorch
    ```



## :scroll: License and Acknowledgement

This project is released under the Apache 2.0 license.<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).


