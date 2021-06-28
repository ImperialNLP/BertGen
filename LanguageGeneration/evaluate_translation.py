######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Use results of testing to evaluate performance of MT translation task
######

import json
import torch
import operator 
import sacrebleu
import unidecode

import sys

# model = "test001_EN_DE"
# filepath = "/experiments/faidon/test/VL-BERT/checkpoints/test_001_start_taskA_epoch001_MT_test2015.json"
# model = "test001_EN_FR"
# filepath = "/experiments/faidon/test/VL-BERT/checkpoints/test_001_start_taskA_epoch001_EN-FR_MT_test_fr.json"
model = sys.argv[1]
filepath = sys.argv[2]
lang = sys.argv[3]

with open(filepath) as json_file:
    data = json.load(json_file)
    correct = 0
    total=0
    captions_dict = {}
    image_dict = {}
    references = []
    hypotheses = []

    #******************************************************
    # Step 1: Read json to dictionaries
    
    # json file to nested dictionary (each caption with all images)
    for p in data:
        # print(p)
        # total += 1
        # if p['word_de_id'] == p['logit']:
        #     correct += 1
        # else:
        # print('***********************')
        # print('source: ', p['caption_en'])
        # remove accents
        # reference = unidecode.unidecode(p['caption_de'])
        if lang == "second":            
            reference = (p['caption_de'].lower())
            # convert ß to ss (equivalent in german)
            hypothesis = p['generated_sentence'].lower()#.replace('ß','ss')
        else:    
            reference = (p['caption_en'].lower())
            # convert ß to ss (equivalent in german)
            hypothesis = p['generated_sentence'].lower()#.replace('ß','ss')
        # print('reference: ', reference)
        # print('generated: ', hypothesis)
        references.append(reference)
        hypotheses.append(hypothesis)

    

    #******************************************************
    # # Step 2: Get BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(bleu.score)

    with open("./checkpoints/generated_text/"+ model+'_ref.txt', 'w') as f:
        for ref in references:
            f.write("%s\n" % ref)
    with open("./checkpoints/generated_text/"+ model+'_hyp.txt', 'w') as f:
        for hyp in hypotheses:
            f.write("%s\n" % hyp)        