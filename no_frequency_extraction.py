#basic implementation of keyword extraction (no frequency)
#i assume this can be folded into a class taking the top parts as fields instantiated when the object is created

from keybert import KeyBERT

#get model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

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