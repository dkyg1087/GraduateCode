"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math
import numpy as np

def buildTrellis(sentence, wCount, indexDict, smoothConst, tagCount, tagList, initProb, transition, hapax):
    trellis= []
    scale= 1000
    mostFreqTag= max(tagCount, key= tagCount.get)
    suffixDict= {"ly":{"ADV": 9*scale},
                "ed":{"VERB": 7*scale, "ADJ": 2*scale},
                "wise":{"ADV": 6*scale, "ADJ": 4*scale},
                "ing":{"VERB": 7*scale, "NOUN": 2*scale},
                "able":{"ADJ": 9*scale},
                "ible":{"ADJ": 9*scale},
                "ship":{"NOUN": 10*scale},
                "ous":{"ADJ": 10*scale},
                "ment":{"NOUN": 10*scale},
                "ion":{"NOUN": 9*scale, "VERB": 1*scale},
                "ness":{"NOUN": 10*scale},
                "ism":{"NOUN": 10*scale},
                "ee":{"NOUN": 8*scale, "VERB": 1*scale, "ADJ": 1*scale}}

    for i in range(len(sentence)):
        newWord= sentence[i]
            
        pair= []
        if i == 0:
            if newWord in wCount:
                for t in indexDict.keys():
                    if t not in wCount[newWord]:
                        scaledConst= scaleSmoothConst(t, smoothConst, hapax)
                        prob= scaledConst/(tagCount[mostFreqTag] + scaledConst*(len(tagCount)+1))
                        temp= (prob, t)
                        # print("Not in wordTags[]", temp)
                        pair.append(temp)
                    else:
                        prob= wCount[newWord][t]
                        temp= ((initProb[indexDict[t]]*prob), t)
                        # print("In wordTags[]", temp)
                        pair.append(temp)
            else:
                modHapax= hapax
                for suffix in list(suffixDict.keys()):
                    if newWord.endswith(suffix):
                        modHapax= suffixDict(suffix)
                
                
                for t in indexDict.keys():
                    scaledConst= scaleSmoothConst(t, smoothConst, modHapax)
                    totalHapaxNum= sum(modHapax.values())+1
                    if t in modHapax:
                        prob = (modHapax.get(t)+scaledConst)/(totalHapaxNum + scaledConst*(len(tagCount)+1))
                    else:
                        prob= scaledConst/(totalHapaxNum + scaledConst*(len(tagCount)+1))
                    
                    temp= (prob, t)
                    # print("Not in wordTags", temp)
                    pair.append(temp)
        else:
            prob= 0
            for t in indexDict.keys():
                
                idx= indexDict[t]
                for j in range(len(indexDict)):
                    prob= -99999999999999999
                    if newWord in wCount:
                        scaledConst= scaleSmoothConst(t, smoothConst, hapax)
                        if t in wCount[newWord]:
                            prob= wCount[newWord][t]
                        else:
                            
                            prob= scaledConst/(tagCount[mostFreqTag] + scaledConst*(len(tagCount)+1))
                    else:
                        modHapax= hapax
                        for suffix in list(suffixDict.keys()):
                            if newWord.endswith(suffix):
                                modHapax= suffixDict[suffix]
                        scaledConst= scaleSmoothConst(t, smoothConst, modHapax)
                        totalHapaxNum= sum(modHapax.values())+1
                        if t in modHapax:
                            prob = (modHapax.get(t)+scaledConst)/(totalHapaxNum + scaledConst*(len(tagCount)+1))
                        else:
                            prob= scaledConst/(totalHapaxNum + scaledConst*(len(tagCount)+1))

                    prevProb= trellis[i-1][idx][0]
                    prob= prevProb+math.log(transition[idx][j])+math.log(prob)
                    temp= (prob, tagList[idx])
                    if idx == 0:
                        # print("idx== 0", temp)
                        pair.append(temp)
                    elif (prob > pair[j][0]):
                        # print("prob > pair[j][0]", temp)
                        pair[j]= temp
        # print(pair) 
        trellis.append(pair)
    return trellis

def backtrackTrellis(trellis, tagList, indexDict):
    pred= []
    tupList= trellis[len(trellis)-1]
    # print(tupList)
    maxIdx= tupList.index((max(tupList)))
    pred.append(tagList[maxIdx])

    prevTag= max(tupList)
    for i in range(len(trellis)-1, 0, -1):
        prevTag= trellis[i-1][indexDict[prevTag[1]]]    
        pred.insert(0, prevTag[1])
    
    pred[0]= max(trellis[0])[1]
    # print(trellis[0])

    return pred

def scaleSmoothConst(t, smoothConst, hapax):
    # totalHapaxNum= sum(hapax.values())+1
    if t in hapax:
        return hapax.get(t)*smoothConst
    else:
        return smoothConst

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    wCount= {}
    tagCount= {}

    index= 0
    indexDict= {}
    
    for s in train:
        for pair in s:
            w, tag = pair

            if tag in tagCount:
                tagCount[tag]+= 1
            else:
                tagCount[tag]= 1
                indexDict[tag]= index
                index+= 1

            if w not in wCount:
                wCount[w]= {}

            if tag in wCount[w]:
                wCount[w][tag]+= 1
            else:
                wCount[w][tag]= 1
    
    hapax= {}
    for w in wCount.keys():
        # print(len(wCount[i].keys()))
        if len(wCount[w].keys()) == 1:
            hapax[list(wCount[w].keys())[0]] = hapax.get(list(wCount[w].keys())[0], 0)+1
            

    smoothConst= 0.00001

    for w in wCount.keys():
        for tag in wCount[w].keys():
            scaledConst= scaleSmoothConst(tag, smoothConst, hapax)
            wCount[w][tag]= (wCount[w][tag] + scaledConst)/(tagCount[tag] + scaledConst*(len(tag)+1))

    # print(index)
    initProb= np.zeros(index)
    transition= np.zeros(shape=(index, index))

    for s in train:
        flag= True
        for i in range(len(s)-1):
            w, tag = s[i]
            tagIdx= indexDict[tag]

            if flag:
                initProb[tagIdx]+= 1
                flag= False

            next= s[i+1][1]
            transition[tagIdx][indexDict[next]]+= 1

    for i in range(len(initProb)):
        initProb[i]= initProb[i]/(len(train)+1)

    for tag, c in tagCount.items():
        prev= indexDict[tag]
        for i in range(len(transition)):
            scaledConst= scaleSmoothConst(tag, smoothConst, hapax)
            transition[prev][i]= (transition[prev][i] + scaledConst)/(c + scaledConst*(len(tagCount)+1))


    tagList= []
    for tag in indexDict.keys():
        tagList.append(tag)

    result= []
    for sentence in test:
        trellis= buildTrellis(sentence, wCount, indexDict, smoothConst, tagCount, tagList, 
                                initProb, transition, hapax)

        if len(trellis) == 0:
            result.append([])
            continue

        pred= backtrackTrellis(trellis, tagList, indexDict)
        result.append(list(zip(sentence, pred)))
    return result
    # return []