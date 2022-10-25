"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter

 
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_counter = Counter()
    
    word_dict = {}
    for sentence in train:
        for word,tag in sentence:
            if not word_dict.get(word,None):
                word_dict[word] = Counter()
            word_dict[word].update([tag])

    tag_counter.update([tup[1] for sentence in train for tup in sentence])
    tag = tag_counter.most_common(1)[0][0]
    result = []
    
    for sentence in test:
        sent = []
        for word in sentence:
            if word_dict.get(word,None):
                sent.append((word,word_dict[word].most_common(1)[0][0]))
            else:
                sent.append((word,tag))
        result.append(sent)
    return result