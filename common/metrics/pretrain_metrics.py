import torch
from .eval_metric import EvalMetric


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class RelationshipAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(RelationshipAccuracy, self).__init__('RelAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['relationship_logits']
            label = outputs['relationship_label']
            # FM edit: change to deal with sigmoid, single output
            # self.sum_metric += float((logits.argmax(dim=1) == label).sum().item())
            self.sum_metric += float(( ((logits>0.5).to(device=logits.device, dtype=torch.float)).squeeze() == label).sum().item())
            self.num_inst += logits.shape[0]


class MLMAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracy, self).__init__('MLMAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits']
            label = outputs['mlm_label']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyWVC(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyWVC, self).__init__('MLMAccWVC', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_wvc']
            label = outputs['mlm_label_wvc']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyAUX(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyAUX, self).__init__('MLMAccAUX', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_aux']
            label = outputs['mlm_label_aux']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()

class MLMAccuracyGlobal(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1, eval_name='default_name'):
        super(MLMAccuracyGlobal, self).__init__('MLMAccuracy'+eval_name, allreduce, num_replicas)
        self.eval_name = eval_name

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_'+self.eval_name]
            label = outputs['mlm_label_'+self.eval_name]
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyDataset1(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyDataset1, self).__init__('MLMAccDataset1', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_dataset1']
            label = outputs['mlm_label_dataset1']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()

class MLMAccuracyDataset2(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyDataset2, self).__init__('MLMAccDataset2', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_dataset2']
            label = outputs['mlm_label_dataset2']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()

class MLMAccuracyDataset3(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyDataset3, self).__init__('MLMAccDataset3', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_dataset3']
            label = outputs['mlm_label_dataset3']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()

            

class MVRCAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MVRCAccuracy, self).__init__('MVRCAccuracy', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mvrc_logits']
            label = outputs['mvrc_label']
            keep = (label.sum(2) - 1.0).abs() < 0.1
            if keep.sum() > 0:
                #FM note: when [keep] is applied it collapsees logits(batch,#RoI,#classes)
                #to logits(#relevant_RoI, #classes)
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep].argmax(dim=1)).sum().item())
                self.num_inst += keep.sum().item()

class MVRCAccuracyGlobal(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1, eval_name='default_name'):
        super(MVRCAccuracyGlobal, self).__init__('MVRCAccuracy'+eval_name, allreduce, num_replicas)
        self.eval_name = eval_name

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mvrc_logits_'+self.eval_name]
            label = outputs['mvrc_label_'+self.eval_name]
            keep = (label.sum(2) - 1.0).abs() < 0.1
            if keep.sum() > 0:
                #FM note: when [keep] is applied it collapsees logits(batch,#RoI,#classes)
                #to logits(#relevant_RoI, #classes)
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep].argmax(dim=1)).sum().item())
                self.num_inst += keep.sum().item()




