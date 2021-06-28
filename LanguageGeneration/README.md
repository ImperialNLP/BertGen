# BERGEN Generation


The code in this directory is used to train and test in a multitask setting on MMT, MT and IC task. Part of the code builds upon the VL-BERT implementation but significant changes have been made to accommodate for language generation in a multimodal and multilingual setting. The LanguageGeneration task can be used for training or testing as explained in the README file of BERTGEN. 

Custom evaluation scripts have also been developed to obtain BLEU statistics after the testing. These are:

1. evaluate_translation.py


Note: According to the options selected in the yaml file, the LanguageGeneration task can be used to perform MT, MMT or IC.
You can choose the appropriate settings in the 'cfgs' in by setting the MODULE option in the yaml files.

- Training:
    - multitask training:  "MODULE: BERTGENMultitaskTraining"
- Inference:
    - Machine Translation:  "MODULE: BERTGENGenerateMMT"
    - Multimodal Machine Translation:  "MODULE: BERTGENGenerateMT"
    - Image Captioning:  "MODULE: BERTGENGenerateImageOnly"



