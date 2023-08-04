#generate groundtruth csv for SK30

import json
from nltk import tokenize
import pandas as pd

file = open('SK30_ground_truth.json')
data = json.load(file)
transcript = data['results']['transcripts'][0]['transcript']
sentences = tokenize.sent_tokenize(transcript)
dataframe = pd.DataFrame(sentences)
dataframe.to_csv('/Users/ashleybao/Documents/GitHub/KeyBERT_extractions/analyzing files/sk30groundtruth.csv')
print(sentences)

file.close()