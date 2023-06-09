#basic implementation of frequency-aware keyword extraction
#i assume this can be folded into a class taking the top parts as fields instantiated when the object is created

from keybert import KeyBERT

#get model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

#dictionary of word: frequency pairs to keep track of frequencies
keywordCounter = {

}

#get the words and not the similarity values
def Extract(lst): #list --> list
    return [item[0] for item in lst]

#increment the word in the frequency dictionary
def updateKeywordCounter(word): #string --> void
    if word in keywordCounter:
        keywordCounter.update({word: keywordCounter[word] + 1})
    else:
        keywordCounter.update({word: 1})

def KeyBERTextract(text, max_frequency): #you may want to move max_frequency into being a field instead of method parameter
    keywords = kw_model.extract_keywords(text, top_n=1000) #default is top_n =5, but i want all keywords
                                                            #this returns a list of (word, similarity) tuples.
    for word in keywords:
        if word[0] in keywordCounter:
            if keywordCounter[word[0]] > max_frequency:
                # will return null if all keywords have reached max frequency
                keywords.remove(word)
    if keywords:
        keyword = Extract(keywords)[0]
        updateKeywordCounter(keyword)
        return keyword
    else:
        return "no keyword" #can change this to null or whatever works with the system.
                            # sometimes, you don't want to recommend a word bc asl wouldn't sign the concept
                            #ex. "the", "because", etc.



