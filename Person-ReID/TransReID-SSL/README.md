![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# TransReID-SSL-DenoiseRep
We apply our proposed [DenoiseRep](../../denoiserep_op/) to [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL/) and get stable improvements.


## Experimental Results
| **Dataset**          | DukeMTMC         | MSMT                   | Market-1501           | CUHK03-L               |
|----------------------|-------------------|------------------------|------------------------|------------------------|
| **TransReID-SSL**         | 81.2%[(model)](https://drive.google.com/file/d/1imEaz1TiPECUddCgqRAgFKdZu1TpGiUO/view?usp=sharing)           | 66.3%[(model)](https://drive.google.com/file/d/1tptfrUg2mxKBwn3PLWU_JMBdbK3NfuQf/view?usp=sharing)                 | 91.2%[(model)](https://drive.google.com/file/d/1o4syrjVyUHRbO7Mgq-xZnjlfEVVufEcF/view?usp=sharing)           | 83.4%[(model)](https://drive.google.com/file/d/12x0pzDrg1GSy1mv-7m_cYnXJAJWHDcE0/view?usp=sharing)                |
| **+_DenoiseRep_** | **82.1%[(model)](https://drive.google.com/file/d/1KKPzh4ABSkrS8n5uLHY9ypZMiO4NmqMo/view?usp=sharing)**    | **67.3%[(model)](https://drive.google.com/file/d/1ZSSylQTMg9zxHukVDBhZevIprybByuMl/view?usp=sharing)**             | **91.9%[(model)](https://drive.google.com/file/d/1-MvY3MWAgHvb_O4r1oDBUNTUnB4hIaFz/view?usp=sharing)**            | **84.6%[(model)](https://drive.google.com/file/d/1YU0ELebkzKpq8FfO7w0ydr4wZX3_66mg/view?usp=sharing)**             |

If you want to train a baseline model yourself, you can refer to [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL/).

## Requirements

### Installation
```bash
pip install -r requirements.txt
```
We recommend to use /torch=1.7.1 /torchvision=0.8.2 /timm=0.3.4 /cuda>10.1 /faiss-gpu=1.7.1/ 24G or 48G RTX3090 for training and evaluation. If you find some packages are missing, please install them manually.


### Prepare Datasets

```bash
mkdir data
```

Download the datasets:
- [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
- [MSMT17](https://arxiv.org/abs/1711.08565)
- [DuckMTMC](https://drive.google.com/file/d/11FxmKe6SZ55DSeKigEtkb-xQwuq6hOkE/view)
- [CUHK-03](https://pan.baidu.com/s/1XM2OFSqCCtX0f6ERiOLMsQ?pwd=4puj)

Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
├── MSMT17
│   └── train
│   └── test
│   └── ..
└── DuckMTMC
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
└── CUHK-03
│   └── cuhk03_release/
│   └── cuhk03_new_protocol_config_detected.mat
│   └── cuhk03_new_protocol_config_labeled.mat
│   └── ..
```

<!-- ## Pre-trained Models
Please download the pre-trained models and put them into your custom file folder.
| Model         | Download |
| :------:      | :------: |
| ViT-S/16      | [link (coming soom)]() |
| ViT-B/16+ICS  | [link (coming soom)]() | -->

## Training

### Load the model after training the baseline pipeline
You need to load the trained model before starting to train DenoiseRep. The model can be downloaded from the table above.
```bash
# transreid_pytorch/train.py:L75
model.load_param("/pretrain_model_path")
```

We utilize 1  GPU for training. Please modify the `MODEL.PRETRAIN_PATH` and `OUTPUT_DIR` in the config file.

```bash
python train.py --config_file configs/market/vit_small.yml
```

You also can speed up training with 4-GPUs training. But the performance may decrease by 0.1~0.2% mAP.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/market/vit_small.yml
```

## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Some examples:**

```bash
# Market
python test.py --config_file configs/market/vit_small.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT 'XXXX/transformer_120.pth'
```


## ReID performance

We have reproduced the performance to verify the reproducibility. The reproduced results may have a gap of about 0.1~0.2% with the numbers.
