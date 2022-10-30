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
    while current is not None:
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
    
    trans_alpha = .000001
    emiss_alpha = .000001
    
    tag_dict = {}
    # emmision propabilites for train data
    
    for sentence in train:
        for word,tag in sentence:
            if not tag_dict.get(tag,None):
                tag_dict[tag] = Counter()
            tag_dict[tag].update([word])

    tags = set(tag_dict.keys())
    
    tag_dict_cache = {}
    for tag in tag_dict.keys():
        tag_dict_cache[tag] = (len(list(tag_dict[tag].elements())),len(list(tag_dict[tag])))
    
    trans_dict = {}
    # transition propabilites for key as the first tag
    
    for sentence in train:
        for word_tag1,word_tag2 in zip(sentence[:-1],sentence[1:]):
            if not trans_dict.get(word_tag1[1],None):
                trans_dict[word_tag1[1]] = Counter()
            tag_dict[word_tag1[1]].update(word_tag2[1])
            
    trans_dict_cache={}
    for tag in trans_dict.keys():
        trans_dict_cache[tag] = (len(list(trans_dict[tag].elements())),len(list(trans_dict[tag])))
    
    cache_dict = {}
    result = []
    for sentence in test:
        #print(sentence)
        past_node_list = []
        current_node_list = []
        for word in sentence:
            #print(word)
            if word == "START":
                past_node_list = []
                current_node = Node(word,"START",0)
                past_node_list.append(current_node)
            elif word == "END":
                max_idx = max(range(len(past_node_list)), key=lambda i: past_node_list[i].prob)
                current_node = Node(word,"END",0,past_node_list[max_idx])
            else:
                current_node_list = []
                for tag in tags:
                    if tag in ["START", "END"]:
                        continue
                    
                    max_prob_node = None
                    for i in range(len(past_node_list)):
                        n = past_node_list[i]
                        if not (tag+'|'+n.tag) in cache_dict:
                            
                            if trans_dict[n.tag][tag] == 0:
                                pt = trans_alpha / (trans_dict_cache[n.tag][0] + (trans_alpha * (trans_dict_cache[n.tag][1]+1)))
                                pt = pt / (len(tags) - trans_dict_cache[n.tag][1] - 2)
                            else:
                                pt = (trans_dict[n.tag][tag] + trans_alpha) / (trans_dict_cache[n.tag][0] + (trans_alpha * (trans_dict_cache[n.tag][1]+1)))

                            pt = math.log(pt)
                            cache_dict[tag+'|'+n.tag] = pt
                        else:
                            pt = cache_dict[tag+'|'+n.tag]

                        if not ("*"+word+'|'+tag) in cache_dict:
                            
                            if tag_dict[tag][word] == 0:
                                pe = emiss_alpha / (tag_dict_cache[tag][0]+ (emiss_alpha *(tag_dict_cache[tag][1]+1)))
                            else:
                                pe = (tag_dict[tag][word] + emiss_alpha) / (tag_dict_cache[tag][0]+ (emiss_alpha *(tag_dict_cache[tag][1]+1)))
                            
                            pe = math.log(pe)
                            cache_dict["*"+word+'|'+tag] = pe
                        else:
                            pe = cache_dict["*"+word+'|'+tag]
                    
                        prob = n.prob+pt+pe
                        if max_prob_node is None or max_prob_node.prob < prob:
                            max_prob_node = Node(word,tag,prob,past_node_list[i])

                    current_node_list.append(max_prob_node)
                past_node_list = current_node_list
                
        tag = backtrack(current_node)
        result.append(tag)
    return result
    