

## Installation
Step 1. Create a conda environment and activate it.
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```
Step 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.
On GPU platforms:
```bash
conda install pytorch torchvision -c pytorch
```
On CPU platforms:
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

Step 3. Install MMEngine and MMCV using MIM.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Step 4.  Install MMDetection.
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
For more detailed installation process, please refer to [Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html).

## Dataset Prepare
Install COCO dataset
```bash
python tools/misc/download_dataset.py --dataset-name coco2017
```
```bash
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   │
```
## Getting Started
#### Training
For example, if you want to train Mask-RNN with swin-T, you can simply run:
```bash
python tools/train.py configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py
```

#### Test and inference
To test the trained model, you can simply run:
```bash
python tools/test.py configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py work_dirs/model_checkpoints
```

