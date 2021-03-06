# Prepare Pre-trained Models
Download pre-trained models and organize them as following:
```
code_root/
└── model/
    └── pretrained_model/
        ├── vl-bert-base-prec.model
        ├── bert-base-multilingual-cased/
        │   ├── vocab.txt (replace with BertGen vocab.txt)
        │   ├── bert_config.json
        │   └── pytorch_model.bin        
        └── resnet101-pt-vgbua-0000.model  
        └── BertGen.model
```


## VL-BERT

Our BERTGEN implementation uses the vl-bert-base-prec model. 

| Model Name         | Download Link    |
| ------------------ | ---------------  |
| vl-bert-base-prec  | [GoogleDrive](https://drive.google.com/file/d/1YBFsyoWwz83VPzbimKymSBxE37gYtfgh/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1SvGbE2cjw8jEGWwSfJBFQQ) |

***Note***: the suffix "prec" means Fast-RCNN is fixed during pre-training and for effeciency the visual features is precomputed using
[bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). 

## BERT & ResNet

Download following pre-trained BERT and ResNet and place them under this folder.

* M-BERT: Files can be downloaded from [HuggingFace](https://huggingface.co/bert-base-multilingual-cased#). Important: replace the vocabulary with `vocab.txt` included in the BERTGen repository.
* ResNet101 pretrained on Visual Genome: 
[GoogleDrive](https://drive.google.com/file/d/1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1_yfZG8VqbWmp5Kr9w2DKGQ) 
(converted from [caffe model](https://www.dropbox.com/s/wqada4qiv1dz9dk/resnet101_faster_rcnn_final.caffemodel?dl=1))

## BertGen

Downloaded the trained BertGen model from the link below:

* BERTGen: [BERTGEN.model](https://zenodo.org/record/5137413/files/final_checkpoint.model) and store under `model/pretrained_model/BERTGen/`
