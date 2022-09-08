# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=10.0, pos_prior=0.8,silently=False):
    print_paramter_vals(laplace,pos_prior)
    good_counter = Counter()
    bad_counter = Counter()
    good_dict = {}
    bad_dict = {}
    for words,label in zip(train_set, train_labels):
        if label == 1:
            good_counter.update(words)
        else:
            bad_counter.update(words)
    yhats = []
    pos_prob = math.log(pos_prior)
    neg_prob = math.log(1-pos_prior)
    for doc in tqdm(dev_set,disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        for word_U in doc:
            word = reader.porter_stemmer.stem(word_U,to_lowercase=False)
            # word = word_U
            prob = good_counter[word]
            if prob != 0:
                if word in good_dict:
                    pos_prob += good_dict[word]
                else:
                    good_prob = math.log((prob+laplace)/(good_counter.total()+laplace*(len(good_counter)+1)))
                    good_dict[word] = good_prob
                    pos_prob += good_prob
            else:
                pos_prob += math.log(laplace/(good_counter.total()+(laplace*(len(good_counter)+1))))
            
            prob = bad_counter[word]
            if prob != 0:
                if word in bad_dict:
                    neg_prob += bad_dict[word]
                else:
                    bad_prob = math.log((prob+laplace)/(bad_counter.total()+laplace*(len(bad_counter)+1)))
                    bad_dict[word] = bad_prob
                    neg_prob += bad_prob
            else:
                neg_prob += math.log(laplace/(bad_counter.total()+laplace*(len(bad_counter)+1)))
        if neg_prob > pos_prob :
            yhats.append(0)
        else :
            yhats.append(1)            
    return yhats





def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0,pos_prior=0.5, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    pos_counter = Counter()
    neg_counter = Counter()
    pos_dict = {}
    neg_dict = {}
    for words,label in zip(train_set,train_labels):
        if label == 1:
            pos_counter.update(words)
        else:
            neg_counter.update(words)
    pos_prob = math.log(pos_prior)
    neg_prob = math.log(1-pos_prior)
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        for i in (len(doc)-1):
            word = reader.porter_stemmer.stem(doc[i],to_lowercase=True) +" "+ reader.porter_stemmer.stem(doc[i]+1,to_lowercase=True)
            count = pos_counter[word]
            if count != 0:
                if word in pos_dict:
                    pos_prob+= pos_dict[word]
                else:
                    pass
                    # TODO
            else:
                pass
                # TODO:                     
    return yhats
