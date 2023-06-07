import stanza
import time

nlp = stanza.Pipeline('en')


def retNoun(text):
    doc = nlp(text)

    listNoun = []
    listVerb = []
    listAdj = []
    listAdv = []
    lst = [];

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == "NOUN":
                listNoun.append(word.text)
            elif word.upos == "VERB":
                listVerb.append(word.text)
            elif word.upos == "ADJ":
                listAdj.append(word.text)
            elif word.upos == "ADV":
                listAdv.append(word.text)

    if listNoun:
        lst.extend(listNoun)
    if listVerb:
        lst.extend(listVerb)
    if listAdj:
        lst.extend(listAdj)
    if listAdv:
        lst.extend(listAdv)
    if lst:
        str = lst[0]

    if lst:
        return str
    else:
        return 'new'

start = time.time()
print(retNoun("He thinks it's scary"))
end = time.time()
print(end-start)