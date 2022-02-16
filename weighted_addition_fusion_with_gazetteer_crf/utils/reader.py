import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from log import logger
from utils.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags
from searchtrie.utils_trie import create_tree, convert_examples_to_features_foronesnt
import pdb
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CoNLLReader(Dataset):
    def __init__(self, max_instances=-1, max_length=50, target_vocab=None, pretrained_dir='',
                 encoder_model='xlm-roberta-large'):
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data, gazetteer=None):
        all_data_dir = os.path.dirname(data)
        all_cached_dir = os.path.join(all_data_dir, 'all_cached/')
        process_file_name = data.split('/')[-1]
        lang = process_file_name.split('_')[0]
        mode = (process_file_name.split('_')[-1]).split('.')[0]

        cached_features_file = os.path.join(all_cached_dir, "cached_{}_{}_{}".format(mode, lang, self._max_length))

        if not os.path.exists(all_cached_dir):
            os.makedirs(all_cached_dir)

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            self.instances = torch.load(cached_features_file)
            instance_idx = len(self.instances)
        else:
            dataset_name = data if isinstance(data, str) else 'dataframe'
            logger.info('Reading file {}'.format(dataset_name))
            instance_idx = 0

            feature_dim, feature_dict, feature_ac_trie = create_tree(self.label_to_id, gazetteer)

            for fields in get_ner_reader(data=data):
                if self._max_instances != -1 and instance_idx > self._max_instances:
                    break

                # print(fields)

                # pdb.set_trace()

                # sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_ = self.parse_line_for_ner(fields=fields)
                sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, feature_input_, head_pos = convert_examples_to_features_foronesnt(
                    fields,
                    self.label_to_id,
                    self._max_length,
                    self.tokenizer,
                    feature_dim,
                    feature_dict,
                    feature_ac_trie,
                    label_mode='a',
                    fea_mode='b')
                # print(sentence_str)
                # print(tokens_sub_rep)
                # print(coded_ner_)

                tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
                tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
                token_masks_rep = torch.tensor(token_masks_rep, dtype=torch.long)

                feature_input = torch.tensor(feature_input_, dtype=torch.float32)

                self.instances.append(
                    (tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, feature_input, head_pos))
                instance_idx += 1

            logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file,
                                                                                       len(self.instances)))
            torch.save(self.instances, cached_features_file)
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), data))
        # pdb.set_trace()

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']

        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)
            ner_tags_rep.extend(tags)

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep = [True] * len(tokens_sub_rep)
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep
