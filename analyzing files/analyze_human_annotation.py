#compare KeyBERT, POS, and GlobalFreq to human annotations of word recommendation

from recStrategyImplementation import strategies
import pandas as pd
import re

dataframe = pd.read_csv("Human Annotation vs. Rec Strategy - Sheet1.csv")
sentences = dataframe['Sentence'].tolist()
Ashley = dataframe['Ashley'].tolist()
Kaleb = dataframe['Kaleb'].tolist()
Ekram = dataframe['Ekram'].tolist()
Madeleine = dataframe['Madeleine'].tolist()
Yifan = dataframe['Yifan'].tolist()
KeyBERT = dataframe['KeyBERT'].dropna().tolist()
POS = dataframe['POS'].dropna().tolist()
GlobalFreq = dataframe['GlobalFreq'].dropna().tolist()
Columns = [Ashley, Kaleb, Ekram, Madeleine, Yifan]

print(KeyBERT)

num_sentences = len(sentences)

for sent in sentences:
    cleaned = re.sub(r'[.,"\'-?:!;]', '', sent)
    KeyBERT.append(strategies.KeyBERTextract(cleaned))
    POS.append(strategies.retNoun(cleaned)[0])
    GlobalFreq.append(strategies.getMostFrequent(cleaned))

print(KeyBERT)
print(POS)
print(GlobalFreq)
print(len(KeyBERT))

def getRates(list):
    KeyBERT_matches = 0
    POS_matches = 0
    GlobalFreq_matches = 0
    for idx, word in enumerate(list, start=0):
        if word.lower() == KeyBERT[idx].lower():
            KeyBERT_matches += 1
        if word.lower() == POS[idx].lower():
            POS_matches += 1
        if word.lower() == GlobalFreq[idx].lower():
            GlobalFreq_matches += 1
    return [KeyBERT_matches, POS_matches, GlobalFreq_matches]

rates = pd.DataFrame()
rates['Ashley_rates'] = getRates(Ashley)
rates['Kaleb_rates'] = getRates(Kaleb)
rates['Ekram_rates'] = getRates(Ekram)
rates['Madeleine_rates'] = getRates(Madeleine)
rates['Yifan_rates'] = getRates(Yifan)

rates.to_csv('out.csv')


