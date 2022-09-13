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

def naiveBayes(train_set, train_labels, dev_set, laplace=13.0, pos_prior=0.8,silently=False):
    print_paramter_vals(laplace,pos_prior)    
    pos_counter = Counter()
    neg_counter = Counter()
    good_dict = {}
    bad_dict = {}
    for words,label in zip(train_set,train_labels):
        if label == 1:
            pos_counter.update(words)
        else:
            neg_counter.update(words)
    pos_prob = math.log(pos_prior)
    neg_prob = math.log(1-pos_prior)
    
    pos_length = len(pos_counter)
    pos_total = pos_counter.total()
    pos_smooth = math.log(laplace/(pos_total+(laplace*(pos_length+1))))
    
    neg_length = len(neg_counter)
    neg_total = neg_counter.total()
    neg_smooth = math.log(laplace/(neg_total+(laplace*(neg_length+1))))
    yhats = []
    # print(pos_smooth,neg_smooth)
    for doc in tqdm(dev_set,disable=silently):
        
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        
        for word in doc:
            prob = pos_counter[word]
            
            if prob != 0:
                good_prob = math.log((prob+laplace)/(pos_total+laplace*(pos_length+1)))
                good_dict[word] = good_prob
                pos_prob += good_prob
            else:
                pos_prob += pos_smooth
            
            prob = neg_counter[word]
            
            if prob != 0:
                bad_prob = math.log((prob+laplace)/(neg_total+laplace*(neg_length+1)))
                bad_dict[word] = bad_prob
                neg_prob += bad_prob
            else:
                neg_prob += neg_smooth
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
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=13, bigram_laplace=0.005, bigram_lambda=0.24,pos_prior=0.8, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    ans_prob = []
    pos_counter = Counter()
    neg_counter = Counter()
    good_dict = {}
    bad_dict = {}
    for words,label in zip(train_set,train_labels):
        if label == 1:
            pos_counter.update([(word1,word2) for word1,word2 in zip(words[:-1],words[1:])])
        else:
            neg_counter.update([(word1,word2) for word1,word2 in zip(words[:-1],words[1:])])
    pos_prob = math.log(pos_prior)
    neg_prob = math.log(1-pos_prior)
    
    pos_length = len(pos_counter)
    pos_total = pos_counter.total()
    pos_smooth = math.log(bigram_laplace/(pos_total+(bigram_laplace*(pos_length+1))))
    
    neg_length = len(neg_counter)
    neg_total = neg_counter.total()
    neg_smooth = math.log(bigram_laplace/(neg_total+(bigram_laplace*(neg_length+1))))

    for doc in tqdm(dev_set,disable=silently):
        
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        
        for word in [(word1,word2) for word1,word2 in zip(doc[:-1],doc[1:])]:
            prob = pos_counter[word]
             
            if prob != 0:
                good_prob = math.log((prob+bigram_laplace)/(pos_total+bigram_laplace*(pos_length+1)))
                good_dict[word] = good_prob
                pos_prob += good_prob
            else:
                pos_prob += pos_smooth
            
            prob = neg_counter[word]
            
            if prob != 0:
                bad_prob = math.log((prob+bigram_laplace)/(neg_total+bigram_laplace*(neg_length+1)))
                bad_dict[word] = bad_prob
                neg_prob += bad_prob
            else:
                neg_prob += neg_smooth
        ans_prob.append((pos_prob,neg_prob))
        '''
        Unigram start here
        ''' 
    pos_counter = Counter()
    neg_counter = Counter()
    good_dict = {}
    bad_dict = {}
    for words,label in zip(train_set,train_labels):
        if label == 1:
            pos_counter.update(words)
        else:
            neg_counter.update(words)
    pos_prob = math.log(pos_prior)
    neg_prob = math.log(1-pos_prior)
    
    pos_length = len(pos_counter)
    pos_total = pos_counter.total()
    if unigram_laplace == 0:
        pos_smooth = 0
    else:
        pos_smooth = math.log(unigram_laplace/(pos_total+(unigram_laplace*(pos_length+1))))
    
    neg_length = len(neg_counter)
    neg_total = neg_counter.total()
    if unigram_laplace == 0:
        neg_smooth = 0
    else:
        neg_smooth = math.log(unigram_laplace/(neg_total+(unigram_laplace*(neg_length+1))))
    i = 0
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        
        for word in doc:
            prob = pos_counter[word]
            
            if prob != 0:
                good_prob = math.log((prob+unigram_laplace)/(pos_total+unigram_laplace*(pos_length+1)))
                good_dict[word] = good_prob
                pos_prob += good_prob
            else:
                pos_prob += pos_smooth
            
            prob = neg_counter[word]
            
            if prob != 0:
                bad_prob = math.log((prob+unigram_laplace)/(neg_total+unigram_laplace*(neg_length+1)))
                bad_dict[word] = bad_prob
                neg_prob += bad_prob
            else:
                neg_prob += neg_smooth
    
        if ans_prob[i][1]* bigram_lambda +  neg_prob * (1-bigram_lambda)> ans_prob[i][0]* bigram_lambda +  pos_prob * (1-bigram_lambda) :
            yhats.append(0)
        else :
            yhats.append(1)
        # print(i)
        i += 1
    print(len(yhats))
    return yhats