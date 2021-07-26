import random
import os
import time
import json
import jsonlines
from PIL import Image
import base64
import numpy as np
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

from copy import deepcopy


class Multi30kDatasetImageOnly(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, seq_len=64,
                 with_precomputed_visual_feat=False, mask_raw_pixels=True,
                 with_rel_task=True, with_mlm_task=True, with_mvrc_task=True,
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, task_name="None", lang="second", **kwargs):
        """
        Conceptual Captions Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(Multi30kDatasetImageOnly, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        # FM edit: commented out to allow testin
        # assert not test_mode

        annot = {'train': 'train_frcnn.json',
                 'val': 'val_frcnn.json',
                 'test2015': 'test_frcnn.json',
                 'test2018': 'test_frcnn2018.json',
                  }

        self.seq_len = seq_len
        self.with_rel_task = with_rel_task
        self.with_mlm_task = with_mlm_task
        self.with_mvrc_task = with_mvrc_task
        self.data_path = data_path
        self.root_path = root_path
        self.ann_file = os.path.join(data_path, annot[image_set])
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.mask_raw_pixels = mask_raw_pixels
        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
                'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
                cache_dir=self.cache_dir, do_lower_case=False)

        self.zipreader = ZipReader()

        # FM: define task name to add prefix
        self.task_name = task_name
        self.lang = lang

        # FM: Customise for multi30k dataset
        self.simple_database = list(jsonlines.open(self.ann_file))
        if not self.zip_mode:
            for i, idb in enumerate(self.simple_database):
                self.simple_database[i]['frcnn'] = idb['frcnn'].replace('.zip@', '')\
                    .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                self.simple_database[i]['image'] = idb['image'].replace(
                    '.zip@', '')

        # FM: TODO correct this
        for i, idb in enumerate(self.simple_database):
            # correct address:
            idb['frcnn'] = idb['frcnn'].replace(
                "test_2016_flickr_frcnn.zip", "test_frcnn.zip")
            old_id = idb['frcnn'].split('/')[1].split('.')[0]
            image_id = old_id
            while len(image_id) < 8:
                image_id = '0'+image_id
            self.simple_database[i]['frcnn'] = idb['frcnn'].replace(
                old_id, image_id)

        if not self.test_mode:
            self.database = []
            db_pos = 0
            # create [MASK] every time
            for entry in self.simple_database:
                if self.lang == "second":
                    caption_tokens_de = self.tokenizer.tokenize(
                        entry['caption_de'])
                    # repeat each entry multiple times - MASK the last word in each case
                    for pos, item in enumerate(caption_tokens_de):
                        self.database.append(deepcopy(entry))
                        self.database[db_pos]['caption_de'] = deepcopy(
                            caption_tokens_de[:pos+1])
                        db_pos += 1
                    # add one last entry with last token [STOP]
                    self.database.append(deepcopy(self.database[db_pos-1]))
                    self.database[db_pos]['caption_de'] = self.database[db_pos]['caption_de'] + ['[STOP]']
                    db_pos += 1
                else:
                    caption_tokens_en = self.tokenizer.tokenize(
                        entry['caption_en'])
                    # repeat each entry multiple times - MASK the last word in each case
                    for pos, item in enumerate(caption_tokens_en):
                        self.database.append(deepcopy(entry))
                        self.database[db_pos]['caption_en'] = deepcopy(
                            caption_tokens_en[:pos+1])
                        db_pos += 1
                    # add one last entry with last token [STOP]
                    self.database.append(deepcopy(self.database[db_pos-1]))
                    self.database[db_pos]['caption_en'] = self.database[db_pos]['caption_en'] + ['[STOP]']
                    db_pos += 1
            print('***********************')
            print('The dataset length is: ', len(self.database))
            print('Task: ', self.task_name)
            print('Lang: ', self.lang)
        else:
            # ignore multiple in turkish (2xcaption), english/german(5xcaption)
            if self.task_name == '[TO_TU]':
                self.database = self.simple_database[::2]
            elif self.task_name == '[TO_FR]':
                self.database = self.simple_database
            else:
                self.database = self.simple_database[::5]

        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

        print('mask_raw_pixels: ', self.mask_raw_pixels)

    @property
    def data_names(self):
        return ['image', 'boxes', 'im_info', 'text',
                'relationship_label', 'mlm_labels', 'mvrc_ops', 'mvrc_labels']

    def __getitem__(self, index):
        idb = self.database[index]

        # image data
        # IN ALL CASES: boxes and cls scores are available for each image
        frcnn_data = self._load_json(
            os.path.join(self.data_path, idb['frcnn']))
        boxes = np.frombuffer(self.b64_decode(frcnn_data['boxes']),
                              dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
        boxes_cls_scores = np.frombuffer(self.b64_decode(frcnn_data['classes']),
                                         dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
        # print('**************')
        # print('boxes shape: ', boxes.shape)
        # print('cls scores shape: ', boxes_cls_scores.shape)
        boxes_max_conf = boxes_cls_scores.max(axis=1)
        inds = np.argsort(boxes_max_conf)[::-1]
        boxes = boxes[inds]
        boxes_cls_scores = boxes_cls_scores[inds]
        boxes = torch.as_tensor(boxes)

        # load precomputed features or the whole image depending on setup
        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = frcnn_data['image_w'], frcnn_data['image_h']
            boxes_features = np.frombuffer(self.b64_decode(frcnn_data['features']),
                                           dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
            boxes_features = boxes_features[inds]
            boxes_features = torch.as_tensor(boxes_features)
        else:
            try:
                image = self._load_image(
                    os.path.join(self.data_path, idb['image']))
                w0, h0 = image.size
            except:
                print("Failed to load image {}, use zero image!".format(
                    idb['image']))
                image = None
                w0, h0 = frcnn_data['image_w'], frcnn_data['image_h']

        # append whole image to tensor of boxes (used for all linguistic tokens)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1.0, h0 - 1.0]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                image_box_feat = boxes_features.mean(dim=0, keepdim=True)
                boxes_features = torch.cat(
                    (image_box_feat, boxes_features), dim=0)

        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(
                image, boxes, None, im_info)

        if image is None and (not self.with_precomputed_visual_feat):
            w = int(im_info[0].item())
            h = int(im_info[1].item())
            image = im_info.new_zeros((3, h, w), dtype=torch.float)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w-1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h-1)

        # Task #1: Caption-Image Relationship Prediction
        _p = random.random()
        if _p < 0.5 or (not self.with_rel_task):
            relationship_label = 1
            if self.lang == "second":
                caption_de = idb['caption_de']
            else:
                caption_en = idb['caption_en']
        else:
            relationship_label = 0
            rand_index = random.randrange(0, len(self.database))
            while rand_index == index:
                rand_index = random.randrange(0, len(self.database))
            if self.lang == "second":
                caption_de = self.database[rand_index]['caption_de']
            else:
                caption_en = self.database[rand_index]['caption_en']

        # Task #2: Masked Language Modeling - Adapted for two languages

        if self.with_mlm_task:
            if not self.test_mode:
                if self.lang == "second":
                    # FM edit: Mask always the last token
                    caption_tokens_de = caption_de
                    mlm_labels_de = [-1] * (len(caption_tokens_de)-1)
                    try:
                        mlm_labels_de.append(
                            self.tokenizer.vocab[caption_tokens_de[-1]])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        mlm_labels_de.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning(
                            "Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
                    caption_tokens_de[-1] = '[MASK]'
                else:
                    # FM edit: Mask always the last token
                    caption_tokens_en = caption_en
                    mlm_labels_en = [-1] * (len(caption_tokens_en)-1)
                    try:
                        mlm_labels_en.append(
                            self.tokenizer.vocab[caption_tokens_en[-1]])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        mlm_labels_en.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning(
                            "Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
                    caption_tokens_en[-1] = '[MASK]'
            else:
                if self.lang == "second":
                    # FM edit: add [MASK] to start guessing caption
                    caption_tokens_de = self.tokenizer.tokenize(caption_de)
                    # FM edit: add label from vocabulary
                    mlm_labels_de = [103] + [-1]
                    caption_tokens_de = ['[MASK]'] + ['[PAD]']
                else:
                    # FM edit: add [MASK] to start guessing caption
                    caption_tokens_en = self.tokenizer.tokenize(caption_en)
                    # FM edit: add label from vocabulary
                    mlm_labels_en = [103] + [-1]
                    caption_tokens_en = ['[MASK]'] + ['[PAD]']
        else:
            if self.lang == "second":
                caption_tokens_de = self.tokenizer.tokenize(caption_de)
                mlm_labels_de = [-1] * len(caption_tokens_de)
            else:
                caption_tokens_en = self.tokenizer.tokenize(caption_en)
                mlm_labels_en = [-1] * len(caption_tokens_en)

        if self.lang == "second":
            text_tokens = [self.task_name] + ['[CLS]'] + \
                ['[SEP]'] + caption_tokens_de + ['[SEP]']
            mlm_labels = [-1] + [-1] + [-1] + mlm_labels_de + [-1]
        else:
            text_tokens = [self.task_name] + ['[CLS]'] + \
                ['[SEP]'] + caption_tokens_en + ['[SEP]']
            mlm_labels = [-1] + [-1] + [-1] + mlm_labels_en + [-1]

        # Task #3: Masked Visual Region Classification
        if self.with_mvrc_task:
            if self.add_image_as_a_box:
                mvrc_ops, mvrc_labels = self.random_mask_region(
                    boxes_cls_scores)
                mvrc_ops = [0] + mvrc_ops
                mvrc_labels = [np.zeros_like(
                    boxes_cls_scores[0])] + mvrc_labels
                num_real_boxes = boxes.shape[0] - 1
                num_masked_boxes = 0
                if self.with_precomputed_visual_feat:
                    boxes_features[0] *= num_real_boxes
                    for mvrc_op, box_feat in zip(mvrc_ops, boxes_features):
                        if mvrc_op == 1:
                            num_masked_boxes += 1
                            boxes_features[0] -= box_feat
                    boxes_features[0] /= (num_real_boxes -
                                          num_masked_boxes + 1e-5)
            else:
                mvrc_ops, mvrc_labels = self.random_mask_region(
                    boxes_cls_scores)
            assert len(mvrc_ops) == boxes.shape[0], \
                "Error: mvrc_ops have length {}, expected {}!".format(
                    len(mvrc_ops), boxes.shape[0])
            assert len(mvrc_labels) == boxes.shape[0], \
                "Error: mvrc_labels have length {}, expected {}!".format(
                    len(mvrc_labels), boxes.shape[0])
        else:
            mvrc_ops = [0] * boxes.shape[0]
            mvrc_labels = [np.zeros_like(boxes_cls_scores[0])] * boxes.shape[0]

        # zero out pixels of masked RoI
        if (not self.with_precomputed_visual_feat) and self.mask_raw_pixels:
            for mvrc_op, box in zip(mvrc_ops, boxes):
                if mvrc_op == 1:
                    x1, y1, x2, y2 = box
                    image[:, int(y1):(int(y2)+1), int(x1):(int(x2)+1)] = 0

        # store labels for masked regions
        mvrc_labels = np.stack(mvrc_labels, axis=0)

        text = self.tokenizer.convert_tokens_to_ids(text_tokens)

        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=1)

        # truncate seq to max len
        if len(text) + len(boxes) > self.seq_len:
            text_len_keep = len(text)
            box_len_keep = len(boxes)
            while (text_len_keep + box_len_keep) > self.seq_len and (text_len_keep > 0) and (box_len_keep > 0):
                if box_len_keep > text_len_keep:
                    box_len_keep -= 1
                else:
                    text_len_keep -= 1
            if text_len_keep < 2:
                text_len_keep = 2
            if box_len_keep < 1:
                box_len_keep = 1
            boxes = boxes[:box_len_keep]
            text = text[:(text_len_keep - 1)] + [text[-1]]
            mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]
            mvrc_ops = mvrc_ops[:box_len_keep]
            mvrc_labels = mvrc_labels[:box_len_keep]

        return image, boxes, im_info, text, relationship_label, mlm_labels, mvrc_ops, mvrc_labels

    # def random_word(self, tokens):
    #     output_label = []
    #
    #     for i, token in enumerate(tokens):
    #         prob = random.random()
    #         # mask token with 15% probability
    #         if prob < 0.15:
    #             prob /= 0.15
    #
    #             # 80% randomly change token to mask token
    #             if prob < 0.8:
    #                 tokens[i] = "[MASK]"
    #
    #             # 10% randomly change token to random token
    #             elif prob < 0.9:
    #                 tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
    #
    #             # -> rest 10% randomly keep current token
    #
    #             # append current token to output (we will predict these later)
    #             try:
    #                 output_label.append(self.tokenizer.vocab[token])
    #             except KeyError:
    #                 # For unknown words (should not occur with BPE vocab)
    #                 output_label.append(self.tokenizer.vocab["[UNK]"])
    #                 logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
    #         else:
    #             # no masking token (will be ignored by loss function later)
    #             output_label.append(-1)
    #
    #     # if no word masked, random choose a word to mask
    #     if self.force_mask:
    #         if all([l_ == -1 for l_ in output_label]):
    #             choosed = random.randrange(0, len(output_label))
    #             output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]
    #
    #     return tokens, output_label

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                # prob /= 0.15
                # FM edit: always leave as mask
                # 80% randomly change token to mask token
                # if prob < 0.8:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                # elif prob < 0.9:
                #     for sub_token in sub_tokens:
                #         output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                #         # -> rest 10% randomly keep current token
                # else:
                #     for sub_token in sub_tokens:
                #         output_tokens.append(sub_token)

                    # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning(
                            "Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        # if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label

    def random_mask_region(self, regions_cls_scores):
        num_regions, num_classes = regions_cls_scores.shape
        output_op = []
        output_label = []
        for k, cls_scores in enumerate(regions_cls_scores):
            prob = random.random()
            # mask region with 15% probability
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.9:
                    # 90% randomly replace appearance feature by "MASK"
                    output_op.append(1)
                else:
                    # -> rest 10% randomly keep current appearance feature
                    output_op.append(0)

                # append class of region to output (we will predict these later)
                output_label.append(cls_scores)
            else:
                # no masking region (will be ignored by loss function later)
                output_op.append(0)
                output_label.append(np.zeros_like(cls_scores))

        # # if no region masked, random choose a region to mask
        # if all([op == 0 for op in output_op]):
        #     choosed = random.randrange(0, len(output_op))
        #     output_op[choosed] = 1
        #     output_label[choosed] = regions_cls_scores[choosed]

        return output_op, output_label

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)
