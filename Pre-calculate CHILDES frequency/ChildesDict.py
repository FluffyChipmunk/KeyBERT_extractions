#create dictionary of lemmas and frequency for CHILDES

import pickle
import pylangacq
import stanza
from multiprocessing import Process, freeze_support, set_start_method
import pandas as pd

distribution = pd.read_csv('distribution.csv')

#lemmas = distribution['lemma'].tolist()

lemma_dict = {

}

for index, row in distribution.iterrows():
    if (row['lemma'], row['pos']) not in lemma_dict:
        lemma_dict.update({(row['lemma'], row['pos']): row['frequency']})
    else:
        lemma_dict.update({(row['lemma'], row['pos']): row['frequency']+lemma_dict[(row['lemma'], row['pos'])]})

with open('../recStrategyImplementation/CHILDES_frequency_data.pkl', 'wb') as fp:
    pickle.dump(lemma_dict, fp)
    print('dictionary saved successfully to file')

with open('../recStrategyImplementation/CHILDES_frequency_data.pkl', 'rb') as fp:
    frequencies = pickle.load(fp)
    print(frequencies[("blue", "ADJ")])




# def getGlobalFrequency(word):
#     return len(childespy.get_types(collection="Eng-NA", token_type=word).index)
#
# # Read dictionary pkl file
# with open('../word_list.pkl', 'rb') as fp:
#     word_list = pickle.load(fp)
#
# print(len(word_list))
#
# CHILDES_frequency = {}
#
#
# for word in word_list:
#     try:
#         CHILDES_frequency.update({word: getGlobalFrequency(word)})
#     except:
#         print("word not available")
#
# print(CHILDES_frequency)
#
