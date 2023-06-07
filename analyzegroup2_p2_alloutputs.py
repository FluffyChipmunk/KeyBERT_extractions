import re

import pandas as pd
from pandas import *
from keybert import KeyBERT
from transformers.pipelines import pipeline
import time

data = read_csv("group2_p2_alloutputs.csv", keep_default_na=False)

Madeleine = data["Madeleine's Recs"].tolist()

keywordRecs = data["keywordRecs"].tolist()

babyRecs = data["babyBERTRecs"].tolist()

mixedRecs = data["mixedRecs"].tolist()

keyphraseRecs = data["keyphraseRecs"].tolist()

currentRecs = data["currentRecs"].tolist()

correctness = {
    "keyword": 0,
    "baby": 0,
    "mixed": 0,
    "keyphrase": 0,
    "current": 0
}


for index, rec in enumerate(Madeleine, start=0):
    if keywordRecs[index] == rec:
        correctness.update({"keyword": correctness["keyword"] + 1})
    if babyRecs[index] == rec:
        correctness.update({"baby": correctness["baby"] + 1})
    if mixedRecs[index] == rec:
        correctness.update({"mixed": correctness["mixed"] + 1})
    if currentRecs[index] == rec:
        correctness.update({"current": correctness["current"] + 1})

totalRecs = 0

for rec in currentRecs:
    if rec != '':
        totalRecs = totalRecs+1

print("total recs:" + str(totalRecs))

print(correctness)


