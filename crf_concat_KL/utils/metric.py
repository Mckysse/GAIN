from collections import defaultdict
from typing import Set
from overrides import overrides

from allennlp.training.metrics.metric import Metric
from reader_utils import get_ner_reader, extract_spans

class SpanF1(Metric):
    def __init__(self, non_entity_labels=['O']) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP, self._FP, self._GT = defaultdict(int), defaultdict(int), defaultdict(int)
        self.non_entity_labels = set(non_entity_labels)

    @overrides
    def __call__(self, batched_predicted_spans, batched_gold_spans, sentences=None):
        non_entity_labels = self.non_entity_labels

        for predicted_spans, gold_spans in zip(batched_predicted_spans, batched_gold_spans):
            gold_spans_set = set([x for x, y in gold_spans.items() if y not in non_entity_labels])
            pred_spans_set = set([x for x, y in predicted_spans.items() if y not in non_entity_labels])

            self._num_gold_mentions += len(gold_spans_set)
            self._num_recalled_mentions += len(gold_spans_set & pred_spans_set)
            self._num_predicted_mentions += len(pred_spans_set)

            for ky, val in gold_spans.items():
                if val not in non_entity_labels:
                    self._GT[val] += 1

            for ky, val in predicted_spans.items():
                if val in non_entity_labels:
                    continue
                if ky in gold_spans and val == gold_spans[ky]:
                    self._TP[val] += 1
                else:
                    self._FP[val] += 1

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        all_tags: Set[str] = set()
        all_tags.update(self._TP.keys())
        all_tags.update(self._FP.keys())
        all_tags.update(self._GT.keys())
        all_metrics = {}
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1_measure = 0.0

        for tag in all_tags:
            precision, recall, f1_measure = self.compute_prf_metrics(true_positives=self._TP[tag],
                                                                     false_negatives=self._GT[tag] - self._TP[tag],
                                                                     false_positives=self._FP[tag])
            all_metrics['P@{}'.format(tag)] = precision
            all_metrics['R@{}'.format(tag)] = recall
            all_metrics['F1@{}'.format(tag)] = f1_measure
            macro_precision += precision
            macro_recall += recall
            macro_f1_measure += f1_measure

            

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self.compute_prf_metrics(true_positives=sum(self._TP.values()),
                                                                 false_positives=sum(self._FP.values()),
                                                                 false_negatives=sum(self._GT.values())-sum(self._TP.values()))
        all_metrics["micro@P"] = precision
        all_metrics["micro@R"] = recall
        all_metrics["micro@F1"] = f1_measure
        
        all_metrics["macro@P"] = macro_precision/len(all_tags)
        all_metrics["macro@R"] = macro_recall/len(all_tags)
        all_metrics["macro@F1"] = macro_f1_measure/len(all_tags)
        
        if self._num_gold_mentions == 0:
            entity_recall = 0.0
        else:
            entity_recall = self._num_recalled_mentions / float(self._num_gold_mentions)

        if self._num_predicted_mentions == 0:
            entity_precision = 0.0
        else:
            entity_precision = self._num_recalled_mentions / float(self._num_predicted_mentions)

        all_metrics['MD@R'] = entity_recall
        all_metrics['MD@P'] = entity_precision
        all_metrics['MD@F1'] = 2. * ((entity_precision * entity_recall) / (entity_precision + entity_recall + 1e-13))
        all_metrics['ALLTRUE'] = self._num_gold_mentions
        all_metrics['ALLRECALLED'] = self._num_recalled_mentions
        all_metrics['ALLPRED'] = self._num_predicted_mentions
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def compute_prf_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    @overrides
    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP.clear()
        self._FP.clear()
        self._GT.clear()

def calculate_macro_f1(label_file, pred_file):
    labels = [field for field in get_ner_reader(label_file)]
    preds = [field for field in get_ner_reader(pred_file)]

    # pdb.set_trace()

    pred_result = []
    label_result = []
    for pred, label in zip(preds, labels):
        label_result.append(extract_spans(label[-1]))
        pred_result.append(extract_spans(pred[-1]))

    span_f1 = SpanF1()
    span_f1(pred_result, label_result)
    word_result = span_f1.get_metric()

    return word_result["macro@F1"]
