import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBertForPretraining
from common.utils.misc import soft_cross_entropy

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class BERTGENMultitaskTraining(Module):
    def __init__(self, config):

        super(BERTGENMultitaskTraining, self).__init__(config)

        # Constructs/initialises model elements
        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(
            1, config.NETWORK.VLBERT.hidden_size)
        if config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding = nn.Embedding(1, 2048)
        if config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding = nn.Embedding(
                1, config.NETWORK.VLBERT.hidden_size)
        self.aux_text_visual_embedding = nn.Embedding(
            1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(
            config.NETWORK.BERT_MODEL_NAME)
        try:
            self.num_datasets = len(config.TRAIN.BATCH_IMAGES)
        except:
            self.num_datasets = 1
        # Can specify pre-trained model or use the downloaded pretrained model specific in .yaml file
        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            # language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
            #                                                           config.NETWORK.BERT_PRETRAINED_EPOCH)
            # FM edit: just use path of pretrained model
            language_pretrained_model_path = config.NETWORK.BERT_PRETRAINED
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(
                config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBertForPretraining(
            config.NETWORK.VLBERT,
            language_pretrained_model_path=None if config.NETWORK.VLBERT.from_scratch else language_pretrained_model_path,
            with_rel_head=config.NETWORK.WITH_REL_LOSS,
            with_mlm_head=config.NETWORK.WITH_MLM_LOSS,
            with_mvrc_head=config.NETWORK.WITH_MVRC_LOSS,
        )

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not self.config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding.weight.data.fill_(0.0)
        if self.config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0,
                                                                std=self.config.NETWORK.VLBERT.initializer_range)
        self.aux_text_visual_embedding.weight.data.normal_(
            mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def train(self, mode=True):
        super(BERTGENMultitaskTraining, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        # In case there were masked values here
        span_tags_fixed = torch.clamp(span_tags, min=0)
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(
            0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def forward(self,
                *args):
                # image,
                # boxes,
                # im_info,
                # text,
                # relationship_label,
                # mlm_labels,
                # mvrc_ops,
                # mvrc_labels,
                # *aux):

        # concat aux texts from different dataset
        # assert len(args) % 8 == 0
        # num_datasets = int(len(args)/8)
        num_datasets = self.num_datasets
        image_list = []
        boxes_list = []
        boxes_mask_list = []
        im_info_list = []
        text_list = []
        relationship_label_list = []
        mlm_labels_list = []
        mvrc_ops_list = []
        mvrc_labels_list = []

        has_visual = []

        max_global_len = 0
        max_global_text_len = 0

        total_examples = 0

        ###########################################
        # Step 1 - Loop through all to get sizes
        ref = 0
        vis_i = 0
        for i in range(num_datasets):
            if args[ref] is None:
                has_visual.append(True)
            else:
                has_visual.append(False)
            if has_visual[i]:
                image_list.append(args[ref])
                boxes_list.append(args[ref+1])
                boxes_mask_list.append((args[ref+1])[:, :, 0] > -1.5)
                im_info_list.append(args[ref+2])
                text_list.append(args[ref+3])
                relationship_label_list.append(args[ref+4])
                mlm_labels_list.append(args[ref+5])
                mvrc_ops_list.append(args[ref+6])
                mvrc_labels_list.append(args[ref+7])

                vis_len = int(boxes_mask_list[vis_i].sum(1).max().item())
                if vis_len > max_global_len:
                    max_global_len = vis_len
                text_len = text_list[i].shape[1]
                if text_len > max_global_text_len:
                    max_global_text_len = text_len
                ref += 8
                vis_i += 1
            else:
                text_list.append(args[ref])
                relationship_label_list.append(args[ref+1])
                mlm_labels_list.append(args[ref+2])

                text_len = text_list[i].shape[1]
                if text_len > max_global_text_len:
                    max_global_text_len = text_len
                ref += 3
            total_examples += text_list[i].shape[0]

        ################################################
        # Step 2 - Loop through datasets
        cur_start = 0
        cur_stop = 0
        vis_i = 0
        box_features_list = []
        obj_reps_list = []
        text_tags_list = []
        text_visual_embeddings_list = []
        object_linguistic_embeddings_list = []
        object_vl_embeddings_list = []

        for i in range(num_datasets):
            if has_visual[i]:
                boxes_mask_list[vis_i] = boxes_mask_list[vis_i][:,
                                                                :max_global_len]
                boxes_list[vis_i] = boxes_list[vis_i][:, :max_global_len]
                mvrc_ops_list[vis_i] = mvrc_ops_list[vis_i][:, :max_global_len]
                mvrc_labels_list[vis_i] = mvrc_labels_list[vis_i][:,
                                                                  :max_global_len]

                if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
                    box_features_list.append(boxes_list[vis_i][:, :, 4:])
                    box_features_list[vis_i][mvrc_ops_list[vis_i] ==
                                             1] = self.object_mask_visual_embedding.weight[0]
                    boxes_list[vis_i][:, :, 4:] = box_features_list[vis_i]

                obj_reps_list.append(self.image_feature_extractor(images=image_list[vis_i],
                                                                  boxes=boxes_list[vis_i],
                                                                  box_mask=boxes_mask_list[vis_i],
                                                                  im_info=im_info_list[vis_i],
                                                                  classes=None,
                                                                  segms=None,
                                                                  mvrc_ops=mvrc_ops_list[vis_i],
                                                                  mask_visual_embed=self.object_mask_visual_embedding.weight[
                                                                      0]
                                                                  if (not self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED)
                                                                  and (not self.config.NETWORK.MASK_RAW_PIXELS)
                                                                  else None))

            ############################################

            # prepare text
            # text_input_ids = text
            # size of sub-batch
            cur_stop += text_list[i].shape[0]
            # creates a text_tags tensor of the same shape as text tensor
            text_tags_list.append(text_list[i].new_zeros(text_list[i].shape))

            if has_visual[i]:
                text_visual_embeddings_list.append(self._collect_obj_reps(
                    text_tags_list[i], obj_reps_list[vis_i]['obj_reps']))

                # linguistic embedding for visual uses [IMG] embedding for all (apart from masked visual)
                object_linguistic_embeddings_list.append(self.object_linguistic_embeddings(
                    boxes_list[vis_i].new_zeros(
                        (boxes_list[vis_i].shape[0], boxes_list[vis_i].shape[1])).long()
                ))
                if self.config.NETWORK.WITH_MVRC_LOSS:
                    object_linguistic_embeddings_list[vis_i][mvrc_ops_list[vis_i]
                                                             == 1] = self.object_mask_word_embedding.weight[0]
                object_vl_embeddings_list.append(torch.cat(
                    (obj_reps_list[vis_i]['obj_reps'], object_linguistic_embeddings_list[vis_i]), -1))

            # Initiliase in first pass
            if i == 0:
                text_input_ids_multi = text_list[i].new_zeros(
                    (total_examples, max_global_text_len))
                text_visual_embeddings_multi = text_visual_embeddings_list[vis_i].new_zeros((total_examples,
                                                                                             max_global_text_len,
                                                                                             text_visual_embeddings_list[vis_i].shape[-1]))
                object_vl_embeddings_multi = object_vl_embeddings_list[vis_i].new_zeros((total_examples, max_global_len,
                                                                                         object_vl_embeddings_list[vis_i].shape[-1]))
                box_mask_multi = boxes_mask_list[vis_i].new_zeros(
                    (total_examples, max_global_len))

            # Concatenates the sub-batches from all dataloaders
            # print("*************")
            # print("list shape: ", text_list[i].shape)
            # print("text_input_ids_multi[cur_start:cur_stop, :text_list[i].shape[1]] shape: ", text_input_ids_multi[cur_start:cur_stop, :text_list[i].shape[1]].shape)
            # print("text_list[i] shape: ", text_list[i].shape)
            text_input_ids_multi[cur_start:cur_stop,
                                 :text_list[i].shape[1]] = text_list[i]
            if has_visual[i]:
                text_visual_embeddings_multi[cur_start:cur_stop, :text_visual_embeddings_list[vis_i].shape[1]] \
                    = text_visual_embeddings_list[vis_i]
                object_vl_embeddings_multi[cur_start:cur_stop,
                                           :object_vl_embeddings_list[vis_i].shape[1], :] = object_vl_embeddings_list[vis_i]
                box_mask_multi[cur_start:cur_stop,
                               :boxes_mask_list[vis_i].shape[1]] = boxes_mask_list[vis_i]

            cur_start = cur_stop
            # TODO: fix to increment if non_visual
            if has_visual[i]:
                vis_i += 1

        # add final
        text_token_type_ids_multi = text_input_ids_multi.new_zeros(
            text_input_ids_multi.shape)
        text_mask_multi = (text_input_ids_multi > 0)

        ###########################################

        # # Visual Linguistic BERT
        # print('text input shape: ', text)
        # print( 'text_input_ids_multi shape: ', text_input_ids_multi.shape)
        # print( 'text_token_type_ids_multi shape: ', text_token_type_ids_multi.shape)
        # print( 'text_visual_embeddings_multi shape: ', text_visual_embeddings_multi.shape)
        # print( 'text_mask_multi shape: ', text_mask_multi.shape)
        # print( 'object_vl_embeddings_multi shape: ', object_vl_embeddings_multi.shape)
        # print( 'box_mask_multi shape: ', box_mask_multi.shape)

        # print ('text_mask_multi: ', text_mask_multi)
        # print ('box_mask_multi: ', box_mask_multi)

        # exit()

        relationship_logits_multi, mlm_logits_multi, mvrc_logits_multi = self.vlbert(text_input_ids_multi,
                                                                                     text_token_type_ids_multi,
                                                                                     text_visual_embeddings_multi,
                                                                                     text_mask_multi,
                                                                                     object_vl_embeddings_multi,
                                                                                     box_mask_multi)

        # print('Logits: ')
        # print('logits shape: ', mlm_logits_multi.shape)
        # exit()

        ###########################################
        ###########################################
        outputs = {}

        # loss
        # relationship_loss = im_info_list.new_zeros(())
        # mlm_loss = im_info_list.new_zeros(())
        # mvrc_loss = im_info.new_zeros(())
        mlm_logits_list = []
        mlm_loss_list = []

        outputs_dict = {}
        mlm_labels_dataset_list = []
        loss = im_info_list[-1].new_zeros(())

        if self.config.NETWORK.WITH_REL_LOSS:
            relationship_logits = relationship_logits_multi[:text_input_ids.shape[0]]
            relationship_loss = F.cross_entropy(
                relationship_logits, relationship_label)
        if self.config.NETWORK.WITH_MLM_LOSS:
            mlm_labels_multi = mlm_labels_list[0].new_zeros((total_examples, max_global_text_len)).fill_(
                -1)

            cur_start = 0
            cur_stop = 0
            for i in range(num_datasets):
                cur_stop += mlm_labels_list[i].shape[0]

                mlm_labels_multi[cur_start:cur_stop,
                                 :mlm_labels_list[i].shape[1]] = mlm_labels_list[i]

                # compute individual losses for reporting metrics
                mlm_loss_list.append(F.cross_entropy(
                    mlm_logits_multi[cur_start:cur_stop].view(
                        (-1, mlm_logits_multi[cur_start:cur_stop].shape[-1])),
                    mlm_labels_multi[cur_start:cur_stop].view(-1),
                    ignore_index=-1
                ))

                # collect data for metrics
                outputs_dict['mlm_logits_' +
                             str(i)] = mlm_logits_multi[cur_start:cur_stop]
                outputs_dict['mlm_label_' +
                             str(i)] = mlm_labels_multi[cur_start:cur_stop]
                outputs_dict['mlm_loss_'+str(i)] = mlm_loss_list[i]

                cur_start = cur_stop

            # USE combined loss for backpropagation - only use per dataset for reporting metrics
            mlm_loss = (F.cross_entropy(
                mlm_logits_multi.view((-1, mlm_logits_multi.shape[-1])),
                mlm_labels_multi.view(-1),
                ignore_index=-1
            ))

            # # calculate total loss

        outputs.update(outputs_dict)
        # outputs.update({
        #     'relationship_logits': relationship_logits if self.config.NETWORK.WITH_REL_LOSS else None,
        #     'relationship_label': relationship_label if self.config.NETWORK.WITH_REL_LOSS else None,
        #     'mlm_logits_wvc': mlm_logits_wvc if self.config.NETWORK.WITH_MLM_LOSS else None,
        #     'mlm_label_wvc': mlm_labels_wvc if self.config.NETWORK.WITH_MLM_LOSS else None,
        #     'mlm_logits_aux': mlm_logits_aux if self.config.NETWORK.WITH_MLM_LOSS else None,
        #     'mlm_label_aux': mlm_labels_aux if self.config.NETWORK.WITH_MLM_LOSS else None,
        #     'mvrc_logits': mvrc_logits if self.config.NETWORK.WITH_MVRC_LOSS else None,
        #     'mvrc_label': mvrc_labels if self.config.NETWORK.WITH_MVRC_LOSS else None,
        #     'relationship_loss': relationship_loss,
        #     'mlm_loss_wvc': mlm_loss_wvc,
        #     'mlm_loss_aux': mlm_loss_aux,
        #     'mvrc_loss': mvrc_loss,
        # })

        loss = mlm_loss.mean()

        return outputs, loss
