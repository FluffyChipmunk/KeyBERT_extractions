#comparing recommendations from multiple "mixed methods"


from pandas import *
from keybert import KeyBERT
import currentRecSystem
import numpy as np
from sklearn.preprocessing import normalize

def truncatedMixExtraction(doc): #round the probabilities of KeyBERT to nearest tenth, cull all but the best words, then use current POS to determine the rest
    keywords = kw_model.extract_keywords(doc, top_n=1000)
    rounded_keywords = []
    for keyword in keywords:
        rounded = (keyword[0], round(keyword[1], 1))
        rounded_keywords.append(rounded)

    partOfSpeechRecs = currentRecSystem.retPrefs(doc)

    max_rounded_keywords = []
    max_rounded_words = []

    max = 0
    for keyword in rounded_keywords:
        if keyword[1]>=max:
            max=keyword[1]
            max_rounded_keywords.append(keyword)

    for word in max_rounded_keywords: #words only
        max_rounded_words.append(word[0])

    if len(max_rounded_keywords)==1:
        return (max_rounded_words[0])
    elif max_rounded_keywords:
        max_partOfSpeechRecs = []
        max = 0
        for rec in partOfSpeechRecs:
            if rec[0] in max_rounded_words and rec[1]>max:
                max = rec[1]
                max_partOfSpeechRecs= rec
        if max_partOfSpeechRecs:
            return max_partOfSpeechRecs[0]
        else:
            return max_rounded_words[0] #just return the first word in max_rounded_words if it doesnt appear in partOfSpeechRecs
    else:
        return "no keyword"

def padKeywords(bigger, smaller):

    padded = []
    smaller_words = []
    for word in smaller:
        smaller_words.append(word[0])
    for word in bigger:
        if word[0] in smaller_words:
            idx = smaller_words.index(word[0])
            padded.append((word[0], smaller[idx][1]))
        else:
            padded.append(("none", 0))
    #print("padded:")
    #print(padded)
    return padded



def weightedExtraction(doc, weight):
    keywords = kw_model.extract_keywords(doc, top_n=1000)
    sorted_keywords = sorted(keywords)
    #print(sorted_keywords)
    partOfSpeechRecs = sorted(currentRecSystem.retPrefs(doc))
    #print(partOfSpeechRecs)

    BERTbigger = True

    if len(sorted_keywords) > len(partOfSpeechRecs):
        bigger = sorted_keywords
        smaller = partOfSpeechRecs
        partOfSpeechRecs = padKeywords(bigger, smaller)
    else:
        bigger = partOfSpeechRecs
        smaller = sorted_keywords
        sorted_keywords = padKeywords(bigger, smaller)
        BERTbigger = False


    BERTwords = []
    BERTvec = []
    for word in sorted_keywords:
        BERTwords.append(word[0])
        BERTvec.append(word[1])
    partOfSpeechWords = []
    partOfSpeechVec = []
    #print(partOfSpeechRecs)
    for word in partOfSpeechRecs:
        #print(word)
        partOfSpeechWords.append(word[0])
        partOfSpeechVec.append(word[1])
    BERTvec = np.array(BERTvec)
    partOfSpeechVec = np.array(partOfSpeechVec)

    #print(BERTwords)
    #print(partOfSpeechWords)

    if len(BERTwords) == len(partOfSpeechWords) and BERTwords:
        BERTvec = normalize([BERTvec])[0]
        partOfSpeechVec = normalize([partOfSpeechVec])[0]

        combined = np.add((1-weight)*BERTvec, weight*partOfSpeechVec)
        combined = normalize([combined])

        if BERTbigger:
            return BERTwords[np.argmax(combined)]
        else:
            return partOfSpeechWords[np.argmax(combined)]
    else:
        return "not the same sizes."


kw_model = KeyBERT()
doc = "Lets' go to the river by the boats."
#print(truncatedMixExtraction(doc))
weight = .5
print(weightedExtraction(doc, weight))

# reading CSV file
data = read_csv("20230405_group2_p2_results - Sheet6.csv", keep_default_na=False)
currentRecs = data['recommendation_lst'].tolist()
globalFreq = read_csv("global frequency no stopwords.csv", keep_default_na=False)['globalFreqRecs'].tolist()
keywordRecs = read_csv("2output_20230405_group2_p2_results.csv", keep_default_na = False)['keywordRecs'].tolist()

options = [currentRecs, globalFreq, keywordRecs]
similarityScores = []
#compare similarity between keyBERT, globalFreq, currentRecs

total = len(keywordRecs)

for strat in options:
    sim =[]
    for strategy in options:
        same =0
        for idx, rec in enumerate(strat):
            if rec.lower() == strategy[idx].lower():
                same = same + 1
        score = same/total
        sim.append(score)
    similarityScores.append(sim)
print(similarityScores)





# converting column data to list
#print(currentRecs)





#global freq is 63% diff from current recs

kw_model = KeyBERT(model ='all-MiniLM-L6-v2' )

transcriptLines = data['transcript_lst'].tolist()
'''globalFrequencyRecs = []

for line in transcriptLines:
    if currentRecs[transcriptLines.index(line)] != '':
        # print("LINE:"+line)
        globalFrequencyRecs.append(currentRecSystem.getMostFrequent(line))
    else:
        globalFrequencyRecs.append('')'''

'''truncatedRecs = []

for line in transcriptLines:
    if currentRecs[transcriptLines.index(line)] != '':
        #print("LINE:"+line)
        truncatedRecs.append(truncatedMixExtraction(line))
    else:
        truncatedRecs.append('')

weight = .2
weighted2 = []
for line in transcriptLines:
    if currentRecs[transcriptLines.index(line)] != '':
        weighted2.append(weightedExtraction(line, weight))
    else:
        weighted2.append('')

weight = .4
weighted4 = []
for line in transcriptLines:
    if currentRecs[transcriptLines.index(line)] != '':
        #print("LINE:"+line)
        weighted4.append(weightedExtraction(line, weight))
    else:
        weighted4.append('')

weight = .6
weighted6 = []
for line in transcriptLines:
    if currentRecs[transcriptLines.index(line)] != '':
        #print("LINE:"+line)
        weighted6.append(weightedExtraction(line, weight))
    else:
        weighted6.append('')

weight = .8
weighted8 = []
for line in transcriptLines:
    if currentRecs[transcriptLines.index(line)] != '':
        #print("LINE:"+line)
        weighted8.append(weightedExtraction(line, weight))
    else:
        weighted8.append('')
'''

#dataframe = DataFrame(transcriptLines)
#dataframe['globalFreqRecs'] = globalFrequencyRecs
'''dataframe['currentRecs'] = currentRecs
dataframe['truncatedRecs'] = truncatedRecs
dataframe['weighted2'] = weighted2
dataframe['weighted4'] = weighted2
dataframe['weighted4'] = weighted4
dataframe['weighted6'] = weighted6
dataframe['weighted8'] = weighted8'''

#dataframe.to_csv('global frequency no stopwords.csv')
