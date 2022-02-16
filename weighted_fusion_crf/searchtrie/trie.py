import logging
import os
import sys
import torch
import pickle
import codecs
import numpy as np
from collections import defaultdict
import pdb
from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrieNode(object):
    def __init__(self, value=None):
        self.value = value
        self.fail = None
        self.tail = 0
        self.children = {}


class Trie(object):
    def __init__(self, words):
        self.root = TrieNode()
        self.count = 0
        self.words = words
        for word in words:
            self.insert(word)
        self.ac_automation()

    def insert(self, sequence):
        self.count += 1
        cur_node = self.root
        for item in sequence:
            if item not in cur_node.children:
                child = TrieNode(value=item)
                cur_node.children[item] = child
                cur_node = child
            else:
                cur_node = cur_node.children[item]

        cur_node.tail = self.count

    def ac_automation(self):
        queue = [self.root]

        while len(queue):
            temp_node = queue[0]
            queue.remove(temp_node)
            for value in temp_node.children.values():
                if temp_node == self.root:
                    value.fail = self.root
                else:
                    p = temp_node.fail
                    while p:
                        if value.value in p.children:
                            value.fail = p.children[value.value]
                            break
                        p = p.fail
                    if not p:
                        value.fail = self.root
                queue.append(value)

    def search_init(self, text):
        p = self.root
        start_index = 0
        rst = defaultdict(list)
        for i in range(len(text)):
            singer_char = text[i]
            while singer_char not in p.children and p is not self.root:
                p = p.fail

            if singer_char in p.children and p is self.root:
                start_index = i  #

            if singer_char in p.children:
                p = p.children[singer_char]
            else:
                start_index = i
                p = self.root
            temp = p
            while temp is not self.root:
                if temp.tail:
                    rst[self.words[temp.tail - 1]].append((start_index, i))
                temp = temp.fail
        return rst

    def search(self, text):
        p = self.root
        rst = []
        for i in range(len(text)):
            singer_char = text[i]
            while singer_char not in p.children and p is not self.root:
                p = p.fail
            if singer_char in p.children:
                p = p.children[singer_char]
            else:
                p = self.root
            temp = p
            while temp is not self.root:
                if temp.tail:
                    theword = self.words[temp.tail - 1]
                    rst.append((i - len(theword) + 1, i))
                temp = temp.fail
        return rst


def build_ac_trie(feature_file):
    try:
        f = codecs.open(feature_file, "r", "utf8")
    except:
        return "open feature file [" + feature_file + "] error!", None
    text_words = []
    try:
        for _, line in enumerate(f):
            line = line.strip()
            linelist = line.split(' ')
            text_words.append(linelist)
        # pdb.set_trace()
        ac_trie = Trie(text_words)
    except:
        return "build ac trie for [" + feature_file + "] error!", None
    return "", ac_trie


def format_query_by_features(text, feature_dim,
                             feature_dict, feature_ac_trie, seq_length):
    if len(text) == 0:
        return "format an empty text", []
    tmp_fea = []
    for i in range(len(text)):
        tmp_fea.append([])

    for fea_name, fea_trie in feature_ac_trie.items():

        feature_list = fea_trie.search(text)

        B_fea = "B-" + fea_name

        I_fea = "I-" + fea_name

        for pos in feature_list:
            # print(pos)
            start = pos[0]
            end = pos[1]
            if start == end:
                tmp_fea[start].append(feature_dict[B_fea])
            elif end - start > 0:
                tmp_fea[start].append(feature_dict[B_fea])
                tmp_fea[end].append(feature_dict[I_fea])
                for i in range(start + 1, end):
                    tmp_fea[i].append(feature_dict[I_fea])
            else:
                return "get feature error", []

    fea = np.ones((seq_length, feature_dim)) * 0

    for idx in range(seq_length):
        if idx > len(text) - 1:
            break
        for fea_idx in range(len(tmp_fea[idx])):
            if tmp_fea[idx][fea_idx] > feature_dim - 1:
                return "feature idx biger than feature dim", []
            fea[idx][tmp_fea[idx][fea_idx]] = 1

    return fea
