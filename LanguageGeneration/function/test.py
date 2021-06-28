######
# Author: Faidon Mitzalis
# Date: June 2020
# Comments: Run model with all caption-image pairs in the dataset
######

import os
import pprint
import shutil

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from LanguageGeneration.data.build import make_dataloader
from LanguageGeneration.modules import *


@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    # ************
    # Step 1: Select model architecture and preload trained model
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(
        ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # ************
    # Step 2: Create dataloader to include all caption-image pairs
    test_loader = make_dataloader(config, mode='test', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # ************
    # Step 3: Run all pairs through model for inference
    generated_sentences = []
    captions_en = []
    captions_de = []
    model.eval()
    cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        # image_ids.extend([test_database[id]['image_index'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        # if 'flickr8k' not in config.DATASET.DATASET_PATH:
        #     captions_en.extend([test_database[id]['caption_en'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        # captions_de.extend([test_database[id]['caption_de'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        batch = to_cuda(batch)
        output = model(*batch)
        generated_sentences.extend((output[0]['generated_sentences']))
        cur_id += bs
        # break
        # exit()
        # TODO: remove this is just for checking
        # if nbatch>900:
        #     break

    # ************
    # Step 3: Store all logit results in file for later evalution
    # if 'flickr8k' not in config.DATASET.DATASET_PATH:
    #     result = [{'generated_sentence': c_id, 'caption_en': caption_en, 'caption_de': caption_de}
    #                 for c_id, caption_en, caption_de in zip(generated_sentences, captions_en, captions_de)]
    # else:
    #     result = [{'generated_sentence': c_id, 'caption_de': caption_de}
    #                 for c_id,  caption_de in zip(generated_sentences, captions_de)]
    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    result_json_path = os.path.join(save_path, '{}.txt'.format(cfg_name if save_name is None else save_name
                                                               ))
    with open(result_json_path, 'w') as f:
        for item in generated_sentences:
            f.write('%s\n' % item)
        # json.dump(result, f)
    print('result json saved to {}.'.format(result_json_path))
    return result_json_path
