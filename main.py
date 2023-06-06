# importing module
import pandas as pd
from pandas import *
from keybert import KeyBERT
from transformers.pipelines import pipeline
hf_model = pipeline("feature-extraction", model="phueb/BabyBERTa-1") #BERT model trained on CHILDES data

#dictionary with words as keys and frequencies as values


def KeyBERTextract(text, model, max_keyphrase_length, num_keywords, max_frequency,keywordCounter):
    kw_model = KeyBERT(model=model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, max_keyphrase_length), top_n=num_keywords)
    for word in keywords:
        if word[0] in keywordCounter:
            if keywordCounter[word[0]] > max_frequency:
                # will return null if all keywords have reached max frequency
                keywords.remove(word)
    if keywords:
        top_word = keywords[0][0]
        if top_word[0] in keywordCounter:
            keywordCounter.update({top_word: keywordCounter[top_word] + 1})
        else:
            keywordCounter.update({top_word: 1})
        return list(zip(*keywords))[0]  # returns the top keyword only (but can see all the other keywords in keywords)
    else:
        return "no key word"

def get_keywordRecs(transcriptLines, model, max_keyphrase_length, num_keywords, max_frequency):
    keywordRecs = []
    keywordCounter = {
    }
    for line in transcriptLines:
        #print(line)
        if currentRecs[transcriptLines.index(line)] != '':
            keywordRecs.append(KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency, keywordCounter))
            # babyBERTRecs.append(KeyBERTextract(line, hf_model, 1, 1, float('inf')))
        else:
            keywordRecs.append('')
    return keywordRecs

def get_mixedRecs(transcriptLines, model, max_keyphrase_length, num_keywords, max_frequency, max_line_length):
    mixedRecs = []
    keywordCounter = {
    }
    for line in transcriptLines:
        #print(line)
        if currentRecs[transcriptLines.index(line)] != '':
            if len(line.split()) <= max_line_length or KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency, keywordCounter) == "no key word":
                mixedRecs.append(currentRecs[transcriptLines.index(line)])
            else:
                mixedRecs.append(KeyBERTextract(line, model, max_keyphrase_length, num_keywords, max_frequency, keywordCounter))
            # babyBERTRecs.append(KeyBERTextract(line, hf_model, 1, 1, float('inf')))
        else:
            mixedRecs.append('')
    return mixedRecs


# reading CSV file
data = read_csv("v2_demo_test_results - nltk_system_results.csv", keep_default_na=False)

# converting column data to list
currentRecs = data['recommendation_lst'].tolist()
print(currentRecs)

#from Uchihara et al meta analysis
frequency = 10

transcriptLines = data['transcript_lst'].tolist()
#keywordRecs = get_keywordRecs(transcriptLines, 'all-MiniLM-L6-v2', 1, 1, frequency)
#mixedRecs =get_mixedRecs(transcriptLines, 'all-MiniLM-L6-v2', 1, 1, frequency, 3)
#babyBERTRecs = []

#for line in transcriptLines:
    #print(line)
    #if currentRecs[transcriptLines.index(line)] != '':
        #keywordRecs.append(KeyBERTextract(line, 'all-MiniLM-L6-v2', 1, 1, float('inf')))
        #babyBERTRecs.append(KeyBERTextract(line, hf_model, 1, 1, float('inf')))
    #else:
        #keywordRecs.append('')
        #babyBERTRecs.append('')
#print(keywordRecs)
dataframe = pd.DataFrame(transcriptLines)
dataframe['currentRecs'] = currentRecs
#dataframe['keywordRecs'] = keywordRecs
#dataframe['mixedRecs'] = mixedRecs

keyphrases = get_keywordRecs(transcriptLines, 'all-MiniLM-L6-v2', 2, 5, frequency)
print(keyphrases)

#dataframe['babyBERTRecs'] = babyBERTRecs
#dataframe.to_csv('output_v2_demo_test_results.csv')
#print(dataframe)


