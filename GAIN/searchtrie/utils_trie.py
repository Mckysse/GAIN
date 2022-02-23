import logging
import os
import numpy as np
import pdb
from torch.nn import CrossEntropyLoss
import copy
from searchtrie.trie import build_ac_trie, format_query_by_features

logger = logging.getLogger(__name__)


def listdir(path, list_name):

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def create_tree(label2id, features_list):
    feature_dim, feature_dict, feature_ac_trie = build_cedar(label2id, features_list)
    return feature_dim, feature_dict, feature_ac_trie


def build_cedar(label2id, features_list):
    feature_dict = label2id
    feature_dim = len(feature_dict)
    feature_set = set()
    errInfo, feature_set = get_feature_names(feature_dict)
    feature_config = {}

    file_path = []
    listdir(features_list, file_path)

    for i in range(len(file_path)):
        position = file_path[i].rfind('/')
        dic_name = file_path[i][position + 1:].replace('.txt', '')
        feature_config[dic_name] = file_path[i]

    feature_ac_trie = {}
    for key, val in feature_config.items():
        if key not in feature_set:
            print("feature names not match")
            print(key)
            continue
        errInfo, feature_ac_trie_sigle = build_ac_trie(val)
        feature_ac_trie[key] = feature_ac_trie_sigle

    return feature_dim, feature_dict, feature_ac_trie


def get_feature_names(feature_dict):
    name_set = set()
    for key in feature_dict.keys():
        feature_split = key.split("-")
        if len(feature_split) == 2:
            name_set.add(feature_split[1])
    return "", name_set


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, ori_tokens, feature):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.ori_tokens = ori_tokens
        self.feature = feature


def add_label_BtoI(label_map, label, len_word):
    word_labels = []
    if label.split('-')[0] == 'B':
        sub_label = 'I-' + label.split('-')[1]
        word_labels.extend([label_map[label]] + [label_map[sub_label]] * (len_word - 1))
    else:
        word_labels.extend([label_map[label]] + [label_map[label]] * (len_word - 1))

    return word_labels


def add_fea_BtoI(label_map, onefea, len_word):
    fea = []
    label2id = label_map
    id2label = {value: key for key, value in label2id.items()}
    nextfea = copy.deepcopy(onefea)
    for pos, if_label in enumerate(onefea):
        if if_label == 0:
            continue
        if if_label == 1:
            labelid_prefix, labelid_class = id2label[pos].split('-')
            if labelid_prefix == 'B':
                nextfea[pos] = 0.0
                nextfea[label2id['I-' + labelid_class]] = 1.0
            elif labelid_prefix == 'I':
                continue

    for i in range(len_word - 1):
        fea.append(nextfea)
    return fea


def extract_spans_byids(tags):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        indicator = nt[0]
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans


def generate_fea_from_labels(labellist, feature_dim, label_map):
    fea = np.ones((len(labellist), feature_dim)) * 0

    for idx, the_label in enumerate(labellist):
        if not (the_label == 'O'):
            fea[idx][label_map[the_label]] = 1

    return fea


def convert_examples_to_features_foronesnt(examples, label_list, max_seq_length, tokenizer, feature_dim, feature_dict,
                                           feature_ac_trie, label_mode='a', fea_mode='a'):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    label_map = {label: i for i, label in enumerate(label_list)}

    textlist = examples[0]
    labellist = examples[-1]
    assert len(textlist) == len(labellist)

    gold_spans_ = extract_spans_byids(labellist)

    fea = format_query_by_features(textlist, feature_dim, feature_dict, feature_ac_trie, len(textlist))
    fea = fea.tolist()

    train_fea = generate_fea_from_labels(labellist, feature_dim, label_map)
    train_fea = train_fea.tolist()

    tokens = []
    label_ids = []
    fea_s = []
    train_fea_s = []
    head_token_pos = []

    # tokenize process
    for word, label, onefea, one_trainfea in zip(textlist, labellist, fea, train_fea):
        word_tokens = tokenizer.tokenize(word)
        if len(word) != 0 and len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        if len(word_tokens) == 0:
            print(textlist)
            raise ValueError("find a None word_token")
        tokens.extend(word_tokens)
        head_token_pos += [1] + [0 for i in range(len(word_tokens) - 1)]

        if label_mode == 'a':
            label_ids.extend(add_label_BtoI(label_map, label, len(word_tokens)))
        elif label_mode == 'b':
            pad_token_label_id = CrossEntropyLoss().ignore_index
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        else:
            raise ValueError("label_mode must be one of a or b")

        if fea_mode == 'a':
            for i in range(len(word_tokens)):
                fea_s.append(onefea)
        elif fea_mode == 'b':
            if len(word_tokens) == 1:
                fea_s.append(onefea)
            elif len(word_tokens) > 1:
                fea_s.append(onefea)
                fea_s.extend(add_fea_BtoI(label_map, onefea, len(word_tokens)))
        else:
            raise ValueError("fea_mode must be one of a or b")

        if fea_mode == 'a':
            for i in range(len(word_tokens)):
                train_fea_s.append(one_trainfea)
        elif fea_mode == 'b':
            if len(word_tokens) == 1:
                train_fea_s.append(one_trainfea)
            elif len(word_tokens) > 1:
                train_fea_s.append(one_trainfea)
                train_fea_s.extend(add_fea_BtoI(label_map, one_trainfea, len(word_tokens)))
        else:
            raise ValueError("fea_mode must be one of a or b")

    # add [CLS] and [SEP]
    if len(tokens) > max_seq_length - 2:
        print('truncate token', len(tokens), max_seq_length, 2)
        tokens = tokens[:(max_seq_length - 2)]
        label_ids = label_ids[:(max_seq_length - 2)]
        fea_s = fea_s[:(max_seq_length - 2)]
        train_fea_s = train_fea_s[:(max_seq_length - 2)]
        head_token_pos = head_token_pos[:(max_seq_length - 2)]

    if label_mode == 'a':
        pad_token_label_id = label_map['O']
    elif label_mode == 'b':
        pad_token_label_id = CrossEntropyLoss().ignore_index
    else:
        raise ValueError("label_mode must be one of a or b")

    tokens = [cls_token] + tokens + [sep_token]
    label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
    zero = [0.] * feature_dim
    fea_s.insert(0, zero)
    fea_s.append(zero)
    head_token_pos = [0] + head_token_pos + [0]

    train_fea_s.insert(0, zero)
    train_fea_s.append(zero)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    assert len(head_token_pos) == len(input_mask)

    padding_length = max_seq_length - len(input_ids)

    input_ids += ([pad_token] * padding_length)
    input_mask += ([0] * padding_length)
    label_ids += ([pad_token_label_id] * padding_length)
    fea_s.extend([zero] * padding_length)
    train_fea_s.extend([zero] * padding_length)

    assert (len(fea_s)) == max_seq_length
    assert (len(train_fea_s)) == max_seq_length

    fea_s = np.array(fea_s)
    train_fea_s = np.array(train_fea_s)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(label_ids) == max_seq_length

    # pdb.set_trace()

    return tokens, input_ids, input_mask, label_ids, gold_spans_, fea_s, train_fea_s, head_token_pos
