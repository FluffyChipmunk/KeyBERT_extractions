import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time



model = SentenceTransformer('all-MiniLM-L6-v2')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data = pd.read_csv("/Users/ashleybao/Documents/GitHub/KeyBERT_extractions/recStrategyImplementation/currentASLdictionary.csv")

asl_gifs = data['name'].tolist() #corpus
asl_words = []
for gif in asl_gifs:
    word = gif.replace('.gif','')
    asl_words.append(word)

corpus_embeddings = model.encode(asl_words, convert_to_tensor=True)

def retrieveWords(keyword):
    query_embedding = model.encode(keyword)
    similar = util.semantic_search(query_embedding, corpus_embeddings, top_k = 1)
    print(similar)
    return asl_words[similar[0][0]['corpus_id']]

print(retrieveWords("cat"))

# #asl_words = ["dog", "cat", "truck", "elephant", "boat"]
# words_tensors = {
# def WordVector(text):
#     return torch.from_numpy(model.encode(text))
# }
#
# for word in asl_words:
#     vec = WordVector(word)
#     words_tensors.update({word: vec})
#
# #print(words_tensors)
# num_words = len(asl_words)
#
# def retrieveWords(keyword):
#     keyword_tensor = WordVector(keyword)
#     compare_tensor = torch.zeros(1, device=device)
#
#
#     q = 0
#     for x in words_tensors:
#         #print(x)
#         cosSim = torch.cosine_similarity(keyword_tensor, words_tensors[x], dim=0)
#         #print("unsqueeze"+(str)(torch.unsqueeze(cosSim, dim=0)))
#         compare_tensor = torch.cat((compare_tensor, (torch.unsqueeze(cosSim, dim=0))))
#         #print(torch.max(torch.cosine_similarity(keyword_tensor, words_tensors[x], dim=0)))
#         q+=1
#     #print("compare_tensor")
#     #print(compare_tensor)
#     #print(max(compare_tensor))
#     index = torch.argmax(compare_tensor).item()
#     #print(index)
#     return asl_words[index-1]
#
# #print(torch.equal(words_tensors["cat"], words_tensors["dog"]))
# #sim = torch.cosine_similarity(words_tensors["dog.gif"], words_tensors["boat.gif"], dim=0)
# #print("sim:")
# #print(sim)
# '''start = time.time()
# print(retrieveWords("vroom"))
# end = time.time()
# print(end-start)
# '''
