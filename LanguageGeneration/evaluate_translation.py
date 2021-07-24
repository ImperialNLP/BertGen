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