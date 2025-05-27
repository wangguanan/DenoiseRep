# Get started: Install and Run MMSeg

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMSegmentation works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

**Note:**
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mmseg python=3.8 -y
conda activate mmseg
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

We recommend that users follow our best practices to install MMSegmentation. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMSegmentation.

Case a: If you develop and run mmseg directly, install it from source:

```shell
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmsegmentation as a dependency or third-party package, install it with pip:

```shell
pip install "mmsegmentation>=1.0.0"
```

## Dataset Prepare
It is recommended to symlink the dataset root to `$MMSEGMENTATION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
```
#### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
We may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

## Training and testing on a single GPU

#### Training on a single GPU

We provide `tools/train.py` to launch training jobs on a single GPU.
The basic usage is as follows.

```shell
python tools/train.py  ${CONFIG_FILE} [optional arguments]
```

This tool accepts several optional arguments, including:

- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--amp`: Use auto mixed precision training.
- `--resume`: Resume from the latest checkpoint in the work_dir automatically.
- `--cfg-options ${OVERRIDE_CONFIGS}`: Override some settings in the used config, and the key-value pair in xxx=yyy format will be merged into the config file.
  For example, '--cfg-option model.encoder.in_channels=6'. Please see this [guide](./1_config.md#Modify-config-through-script-arguments) for more details.

Below are the optional arguments for the multi-gpu test:

- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.

**Note:** Difference between the argument `--resume` and the field `load_from` in the config file:

`--resume` only determines whether to resume from the latest checkpoint in the work_dir. It is usually used for resuming the training process that is interrupted accidentally.

`load_from` will specify the checkpoint to be loaded and the training iteration starts from 0. It is usually used for fine-tuning.

If you would like to resume training from a specific checkpoint, you can use:

```python
python tools/train.py ${CONFIG_FILE} --resume --cfg-options load_from=${CHECKPOINT}
```

**Training on CPU**: The process of training on the CPU is consistent with single GPU training if a machine does not have GPU. If it has GPUs but not wanting to use them, we just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run the script [above](#training-on-a-single-gpu).

### Testing on a single GPU

We provide `tools/test.py` to launch training jobs on a single GPU.
The basic usage is as follows.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

This tool accepts several optional arguments, including:

- `--work-dir`: If specified, results will be saved in this directory. If not specified, the results will be automatically saved to `work_dirs/{CONFIG_NAME}`.
- `--show`: Show prediction results at runtime, available when `--show-dir` is not specified.
- `--show-dir`: Directory where painted images will be saved. If specified, the visualized segmentation mask will be saved to the `work_dir/timestamp/show_dir`.
- `--wait-time`: The interval of show (s), which takes effect when `--show` is activated. Default to 2.
- `--cfg-options`:  If specified, the key-value pair in xxx=yyy format will be merged into the config file.
- `--tta`: Test time augmentation option.

**Testing on CPU**: The process of testing on the CPU is consistent with single GPU testing if a machine does not have GPU. If it has GPUs but not wanting to use them, we just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

then run the script [above](#testing-on-a-single-gpu).

## Training and testing on multiple GPUs and multiple machines

### Training on multiple GPUs

OpenMMLab2.0 implements **distributed** training with `MMDistributedDataParallel`.
We provide `tools/dist_train.sh` to launch training on multiple GPUs.

The basic usage is as follows:

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments remain the same as stated [above](#training-on-a-single-gpu) and have additional arguments to specify the number of GPUs.

An example:

```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512/
# If work_dir is not set, it will be generated automatically.
sh tools/dist_train.sh configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py 8 --work-dir work_dirs/fcn_r50-d8_4xb4-80k_ade20k-512x512
```

**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. A custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use a symlink, for example:

```shell
ln -s ${YOUR_WORK_DIRS} ${MMSEG}/work_dirs
```

### Testing on multiple GPUs

We provide `tools/dist_test.sh` to launch testing on multiple GPUs.
The basic usage is as follows.

```shell
sh tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments remain the same as stated [above](#testing-on-a-single-gpu) and have additional arguments to specify the number of GPUs.

An example:

```shell
./tools/dist_test.sh configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py \
    checkpoints/fcn_r50-d8_4xb4-80k_ade20k-512x512.pth 4
```
