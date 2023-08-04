from keybert import KeyBERT
import stanza
import childespy as childes
import pickle


#get model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

nlp = stanza.Pipeline(lang="en")
ENGLISH_STOP_WORDS = {'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all',
                      'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst',
                      'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway',
                      'anywhere', 'are', 'aren', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become',
                      'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside',
                      'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot',
                      'cant', 'co', 'con', 'could', 'couldn', 'couldnt', 'cry', 'd', 'de', 'describe', 'detail', 'did',
                      'didn', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg',
                      'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever',
                      'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill',
                      'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from',
                      'front', 'full', 'further', 'get', 'give', 'go', 'had', 'hadn', 'has', 'hasn', 'hasnt', 'have',
                      'haven', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
                      'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in',
                      'inc', 'indeed', 'interest', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'keep', 'last',
                      'latter', 'latterly', 'least', 'less', 'll', 'ltd', 'm', 'ma', 'made', 'many', 'may', 'me',
                      'meanwhile', 'might', 'mightn', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move',
                      'much', 'must', 'mustn', 'my', 'myself', 'name', 'namely', 'needn', 'neither', 'never',
                      'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
                      'nowhere', 'o', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
                      'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps',
                      'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems',
                      'serious', 'several', 'shan', 'she', 'should', 'shouldn', 'show', 'side', 'since', 'sincere',
                      'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
                      'somewhere', 'still', 'such', 'system', 't', 'take', 'ten', 'than', 'that', 'the', 'their',
                      'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
                      'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though',
                      'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
                      'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 've', 'very',
                      'via', 'was', 'wasn', 'we', 'well', 'were', 'weren', 'what', 'whatever', 'when', 'whence',
                      'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
                      'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
                      'with', 'within', 'without', 'won', 'would', 'wouldn', 'y', 'yet', 'you', 'your', 'yours',
                      'yourself', 'yourselves'}
#get the words and not the similarity values
def Extract(lst): #list --> list
    return [item[0] for item in lst]
def KeyBERTextract(text): #string --> string
    keywords = kw_model.extract_keywords(text, top_n=1000) #default is top_n =5, but i want all keywords. this returns a list of (word, similarity) tuples.
    if keywords:
        keyword = Extract(keywords)[0]
        return keyword
    else:
        return "no keyword" #can change this to null or whatever works with the system.
                            # sometimes, you don't want to recommend a word bc asl wouldn't sign the concept
                            #ex. "the", "because", etc.

def getGlobalFrequency(word):
    return len(childes.get_types(collection = "Eng-NA", token_type=word).index)


def retNoun(text):
    doc = nlp(text)

    listNoun = []
    listVerb = []
    listAdj = []
    listAdv = []
    lst = [];

    for sent in doc.sentences:
        for word in sent.words:
            #if word not in ENGLISH_STOP_WORDS:
                if word.upos == "NOUN":
                    listNoun.append((word.text, 4))
                elif word.upos == "VERB":
                    listVerb.append((word.text, 3))
                elif word.upos == "ADJ":
                    listAdj.append((word.text, 2))
                elif word.upos == "ADV":
                    listAdv.append((word.text, 1))

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

with open('CHILDES_frequency_data.pkl', 'rb') as fp:
    frequencies = pickle.load(fp)


doc = nlp('The boat is in the water. I love the boat!')
latest = doc.sentences[-1].words
content_words = [word for word in latest
                 if word.text.lower() not in ENGLISH_STOP_WORDS and word.lemma != "be"]
max_frequency = 0
rec_word = ''
for word in content_words:
    if (word.lemma, word.upos) in frequencies:
        frequency = frequencies[(word.lemma, word.upos)]
    else:
        frequency = 0
    if frequency > max_frequency:
        max_frequency = frequency
        rec_word = word
print(rec_word)


def getMostFrequent(text):
    print(text)
    doc = nlp(text)

    listNoun = []
    listVerb = []
    listAdj = []
    listAdv = []
    lst = []

    for sent in doc.sentences:
        for word in sent.words:
            if (word.lemma, word.upos) in frequencies and word.text.lower() not in ENGLISH_STOP_WORDS and word.lemma != "be":
                print(word)
                if word.upos == "NOUN":
                    listNoun.append((word.text, frequencies[(word.lemma, word.upos)]))
                elif word.upos == "VERB":
                    listVerb.append((word.text, frequencies[(word.lemma, word.upos)]))
                elif word.upos == "ADJ":
                    listAdj.append((word.text, frequencies[(word.lemma, word.upos)]))
                elif word.upos == "ADV":
                    listAdv.append((word.text, frequencies[(word.lemma, word.upos)]))

    if listNoun:
        lst.extend(listNoun)
    if listVerb:
        lst.extend(listVerb)
    if listAdj:
        lst.extend(listAdj)
    if listAdv:
        lst.extend(listAdv)

    if lst:
        contentWords = list(set(lst))
        maxFreq = 0
        maxWord = ""
        for word in contentWords:
            if word[1] > maxFreq:
                maxFreq = word[1]
                maxWord = word[0]
        return maxWord
    else:
        #return ''list(set(lst))''
        return ''


#start = time.time()
sentences = ["It is yellow."]
for sentence in sentences:
    print(getMostFrequent(sentence))
#end = time.time()
#print(end-start)

