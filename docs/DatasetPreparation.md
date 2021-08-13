# Dataset Preparation

[English](DatasetPreparation.md) 


## Data Storage Format

At present, there are three types of data storage formats supported:

1. Store in `hard disk` directly in the format of images / video frames.
1. Make [LMDB](https://lmdb.readthedocs.io/en/release/), which could accelerate the IO and decompression speed during training.

#### How to Use

At present, we can modify the configuration yaml file to support different data storage formats. Taking [PairedImageDataset](../basicsr/data/paired_image_dataset.py) as an example, we can modify the yaml file according to different requirements.

1. Directly read disk data.

    ```yaml
    type: VideoTestDataset
    dataroot_gt: ./train_sharp
    dataroot_lq: ./train_sharp_bicubic/X4/
    io_backend:
      type: disk
    ```

1. Use LMDB.
We need to make LMDB before using it. Please refer to [LMDB description](#LMDB-Description). Note that we add meta information to the original LMDB, and the specific binary contents are also different. Therefore, LMDB from other sources can not be used directly.

    ```yaml
    type: REDSDataset
    dataroot_gt: /cluster/work/cvl/videosr/REDS/train_sharp_with_val.lmdb 
    dataroot_lq: /cluster/work/cvl/videosr/REDS/train_sharp_bicubic_with_val.lmdb 
    io_backend:
      type: lmdb
    ```



#### How to Implement

The implementation is to call the elegant fileclient design in [mmcv](https://github.com/open-mmlab/mmcv). In order to be compatible with BasicSR, we have made some changes to the interface (mainly to adapt to LMDB). See [file_client.py](../basicsr/utils/file_client.py) for details.

When we implement our own dataloader, we can easily call the interfaces to support different data storage forms. Please refer to [PairedImageDataset](../basicsr/data/paired_image_dataset.py) for more details.

#### LMDB Description

During training, we use LMDB to speed up the IO and CPU decompression. (During testing, usually the data is limited and it is generally not necessary to use LMDB). The acceleration depends on the configurations of the machine, and the following factors will affect the speed:

1. Some machines will clean cache regularly, and LMDB depends on the cache mechanism. Therefore, if the data fails to be cached, you need to check it. After the command `free -h`, the cache occupied by LMDB will be recorded under the `buff/cache` entry.
1. Whether the memory of the machine is large enough to put the whole LMDB data in. If not, it will affect the speed due to the need to constantly update the cache.
1. If you cache the LMDB dataset for the first time, it may affect the training speed. So before training, you can enter the LMDB dataset directory and cache the data by: ` cat data.mdb > /dev/nul`.

In addition to the standard LMDB file (data.mdb and lock.mdb), we also add `meta_info.txt` to record additional information.
Here is an example:

**Folder Structure**

```txt
DIV2K_train_HR_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**meta information**

`meta_info.txt`, We use txt file to record for readability. The contents are:

```txt
0001_s001.png (480,480,3) 1
0001_s002.png (480,480,3) 1
0001_s003.png (480,480,3) 1
0001_s004.png (480,480,3) 1
...
```

Each line records an image with three fields, which indicate:

- Image name (with suffix): 0001_s001.png
- Image size: (480, 480,3) represents a 480x480x3 image
- Other parameters (BasicSR uses cv2 compression level for PNG): In restoration tasks, we usually use PNG format, so `1` represents the PNG compression level `CV_IMWRITE_PNG_COMPRESSION` is 1. It can be an integer in [0, 9]. A larger value indicates stronger compression, that is, smaller storage space and longer compression time.

**Binary Content**

For convenience, the binary content stored in LMDB dataset is encoded image by cv2: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. You can control the compression level by `compress_level`, balancing storage space and the speed of reading (including decompression).

**How to Make LMDB**
We provide a script to make LMDB. Before running the script, we need to modify the corresponding parameters accordingly. At present, we support DIV2K, REDS and Vimeo90K datasets; other datasets can also be made in a similar way.<br>
 `python scripts/data_preparation/create_lmdb.py`

#### Data Pre-fetcher

Apar from using LMDB for speed up, we could use data per-fetcher. Please refer to [prefetch_dataloader](../basicsr/data/prefetch_dataloader.py) for implementation.<br>
It can be achieved by setting `prefetch_mode` in the configuration file. Currently, it provided three modes:

1. None. It does not use data pre-fetcher by default. If you have already use LMDB or the IO is OK, you can set it to None.

    ```yml
    prefetch_mode: ~
    ```

1. `prefetch_mode: cuda`. Use CUDA prefetcher. Please see [NVIDIA/apex](https://github.com/NVIDIA/apex/issues/304#) for more details. It will occupy more GPU memory. Note that in the mode. you must also set `pin_memory=True`.

    ```yml
    prefetch_mode: cuda
    pin_memory: true
    ```

1. `prefetch_mode: cpu`. Use CPU prefetcher, please see [IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#) for more details. (In my tests, this mode does not accelerate)

    ```yml
    prefetch_mode: cpu
    num_prefetch_queue: 1  # 1 by default
    ```



## Video Super-Resolution

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### REDS

[Official website](https://seungjunnah.github.io/Datasets/reds.html).<br>
We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. And we  rename the validation clips from 240 to 269.

**Validation Partition**

The official validation partition and that used in EDVR for competition are different:

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

All the left clips are used for training. Note that it it not required to explicitly separate the training and validation datasets; and the dataloader does that.

**Preparation Steps**

1. Download the datasets from the [official website](https://seungjunnah.github.io/Datasets/reds.html).
1. Regroup the training and validation datasets: `python scripts/data_preparation/regroup_reds_dataset.py`
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_reds` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_reds_dataset.py`.
Remember to modify the paths and configurations accordingly.

### Vimeo90K

[Official webpage](http://toflow.csail.mit.edu/)

1. Download the dataset: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).This is the Ground-Truth (GT). There is a `sep_trainlist.txt` file listing the training samples in the download zip file.
1. Generate the low-resolution images (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_vimeo90k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_vimeo90k_dataset.py`.
Remember to modify the paths and configurations accordingly.

