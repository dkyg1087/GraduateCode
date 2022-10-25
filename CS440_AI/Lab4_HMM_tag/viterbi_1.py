"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

#THIS IS A TEST VERSION (NOT DONE YET)

import numpy as np
import math
from collections import Counter

class Node:
    def __init__(self,word,tag,prob=0,previous=None):
        self.word = word
        self.tag = tag
        self.prob = prob
        self.previous = previous

def backtrack(end_node:Node):
    current = end_node
    result = []
    while current.previous is not None:
        result.append((current.word,current.tag))
        current = current.previous
    result.reverse()
    return result

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_dict = {}
    
    # emmision propabilites for train data
    
    for sentence in train:
        for word,tag in sentence:
            if not tag_dict.get(tag,None):
                tag_dict[tag] = Counter()
            tag_dict[tag].update([word])

    trans_dict = {}
    
    for sentence in train:
        for word_tag1,word_tag2 in zip(sentence[:-1],sentence[1:]):
            if not trans_dict.get(word_tag1[1],None):
                trans_dict[word_tag1[1]] = Counter()
            tag_dict[word_tag1[1]].update(word_tag2[1])

    cache_dict = {}
    result = []
    for sentence in test:
        sent = []
        for word in sentence:
            if word == "START":
                current_node = Node(word,"START")
            if word == "END":
                
                current_node = Node(word,"END",0,)
            else:
                pass
        result.append(backtrack(current_node))
    return result
    