# importing module
import pandas as pd
from pandas import *
from keybert import KeyBERT
from transformers.pipelines import pipeline
import time
hf_model = pipeline("feature-extraction", model="phueb/BabyBERTa-1") #BERT model trained on CHILDES data


#USED FOR TRANSCRIPTIONS ONLY

#dictionary with words as keys and frequencies as values
kw_model = KeyBERT()

start = time.time()
kw_model.extract_keywords("this is a cat", top_n=1)
end = time.time()
print(end-start)

def Extract(lst):
    return [item[0] for item in lst]

def KeyBERTextract(text, model, max_keyphrase_length, num_keywords, max_frequency,keywordCounter):
    kw_model = KeyBERT(model=model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, max_keyphrase_length), top_n=num_keywords)
    for word in keywords:
        if word[0] in keywordCounter:
            if keywordCounter[word[0]] > max_frequency:
                # will return null if all keywords have reached max frequency
                keywords.remove(word)
    if keywords:
        #print(Extract(keywords))
        return Extract(keywords)
        #return list(list(zip(*keywords)))[0]   #returns the topn keywords
    else:
        return "no key word"
#need to fix the frequency counter for other methods

def updateKeywordCounter(keywordCounter, word):
    if word in keywordCounter:
        keywordCounter.update({word: keywordCounter[word] + 1})
    else:
        keywordCounter.update({word: 1})


def get_keywordRecs(transcriptLines, model, max_keyphrase_length, num_keywords, max_frequency):
    keywordRecs = []
    keywordCounter = {
    }
    for line in transcriptLines:
        #print(line)
        if True:
        #if currentRecs[transcriptLines.index(line)] != '':
            currentKeywords = KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency, keywordCounter)
            for word in currentKeywords:
                updateKeywordCounter(keywordCounter, word)
            keywordRecs.append(currentKeywords)
            # babyBERTRecs.append(KeyBERTextract(line, hf_model, 1, 1, float('inf')))
        else:
            keywordRecs.append('')
    return keywordRecs

'''def get_mixedRecs(transcriptLines, model, max_keyphrase_length, num_keywords, max_frequency, max_line_length):
    mixedRecs = []
    keywordCounter = {
    }
    for line in transcriptLines:
        #print(line)
        if currentRecs[transcriptLines.index(line)] != '':
            if len(line.split()) <= max_line_length or KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency, keywordCounter) == "no key word":
                mixedRecs.append(currentRecs[transcriptLines.index(line)])
            else:
                currentKeywords = KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency,
                                                 keywordCounter)
                for word in currentKeywords:
                    updateKeywordCounter(keywordCounter, word)
                mixedRecs.append(currentKeywords)
            # babyBERTRecs.append(KeyBERTextract(line, hf_model, 1, 1, float('inf')))
        else:
            mixedRecs.append('')
    return mixedRecs'''


'''def get_keyphraseRecs(transcriptLines, model, max_keyphrase_length, num_keywords, max_frequency):
    keywordRecs = []
    keywordCounter = {
    }
    for line in transcriptLines:
        if currentRecs[transcriptLines.index(line)] != '':
            keywordRecs.append(KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency, keywordCounter))
        else:
            keywordRecs.append('')
    return keywordRecs

#def retrieve_keyphrase(keyphraseRecs)
    #for keyphrase in keyphraseRecs
        #for sign in signbank
            #if cos sim > .8: return keyphrase
            #else return get_keywordRecs(transcriptLines, 'all-MiniLM-L6-v2', 1, 1, frequency)[0]'''


''''# reading CSV file
data = read_csv("20230405_group2_p2_results - Sheet6.csv", keep_default_na=False)

# converting column data to list
currentRecs = data['recommendation_lst'].tolist()
print(currentRecs)

#from Uchihara et al meta analysis
frequency = 10

#more realistic testing
#frequency = 3

transcriptLines = data['transcript_lst'].tolist()
keywordRecs = get_keywordRecs(transcriptLines, 'all-MiniLM-L6-v2', 1, 1, frequency)
mixedRecs =get_mixedRecs(transcriptLines, 'all-MiniLM-L6-v2', 1, 1, frequency, 3)
keyphrases = get_keywordRecs(transcriptLines, 'all-MiniLM-L6-v2', 2, 1, frequency)

#babyBERTRecs = get_keywordRecs(transcriptLines, hf_model, 1, 1, frequency)


#print(keywordRecs)
dataframe = pd.DataFrame(transcriptLines)
dataframe['currentRecs'] = currentRecs
dataframe['keywordRecs'] = keywordRecs
dataframe['mixedRecs'] = mixedRecs
dataframe['keyphraseRecs'] = keyphrases

print(keyphrases)

#dataframe['babyBERTRecs'] = babyBERTRecs
dataframe.to_csv('output_20230405_group2_p2_results.csv')
#print(dataframe)
'''

